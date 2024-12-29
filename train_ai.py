from config import (
    GAME_CONFIG, 
    TRAINING_CONFIG, 
    NETWORK_CONFIG,
    LOGGING_CONFIG
)
from minesweeper_ai import MinesweeperAI
from main import Minesweeper
from logger_config import setup_logger
import os
from datetime import datetime
from tkinter import Tk, filedialog
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from multiprocessing.pool import ThreadPool
import torch

class AITrainer:
    def __init__(self):
        self.logger = setup_logger('ai_trainer')
        self.ai = MinesweeperAI(
            GAME_CONFIG['WIDTH'],
            GAME_CONFIG['HEIGHT'],
            GAME_CONFIG['NUM_MINES']
        )
        
        # Создаем директорию для сохранения моделей
        if not os.path.exists(LOGGING_CONFIG['SAVE_DIR']):
            os.makedirs(LOGGING_CONFIG['SAVE_DIR'])
        
        self.best_reward = float('-inf')
        self.best_model_path = None
        self.best_win_rate = 0
        
        # Загружаем существующую модель, если это указано в конфиге
        if TRAINING_CONFIG['LOAD_EXISTING_MODEL']:
            self.load_existing_model()
        
        # Добавляем предварительное выделение памяти
        self.preallocated_games = [
            Minesweeper(
                GAME_CONFIG['WIDTH'],
                GAME_CONFIG['HEIGHT'],
                GAME_CONFIG['NUM_MINES']
            ) for _ in range(TRAINING_CONFIG['BATCH_SIZE'])
        ]
        
        # Добавляем параллельную обработку
        self.num_workers = os.cpu_count()
        self.pool = ThreadPool(self.num_workers)
    
    def load_existing_model(self):
        """Загружает существующую модель через диалог выбора файла"""
        self.logger.info("Выбор существующей модели для продолжения обучения...")
        
        try:
            # Создаем скрытое окно Tkinter
            root = Tk()
            root.withdraw()
            
            # Получаем путь к директории с моделями
            models_dir = os.path.abspath(LOGGING_CONFIG['SAVE_DIR'])
            
            # Открываем диалог выбора файла
            model_path = filedialog.askopenfilename(
                initialdir=models_dir,
                title="Выберите модель для продолжения обучения",
                filetypes=(("PyTorch models", "*.pth"), ("all files", "*.*"))
            )
            
            if model_path:
                self.ai.load_model(model_path)
                self.logger.info(f"Загружена модель: {os.path.basename(model_path)}")
            else:
                self.logger.info("Выбор модели отменен. Используется новая модель.")
        
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            self.logger.info("Используется новая модель.")
    
    def save_if_better(self, avg_reward: float):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_model_path = os.path.join(
                LOGGING_CONFIG['SAVE_DIR'],
                f'model_best_reward_{avg_reward:.3f}.pth'
            )
            self.ai.save_model(self.best_model_path)
    
    def save_best_model(self, win_rate: float):
        """
        Сохраняет модель, если она показала лучший результат
        
        Args:
            win_rate: текущий процент побед
        """
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            
            # Создаем имя файла с процентом побед
            model_name = f'model_best_winrate_{win_rate:.3f}.pth'
            save_path = os.path.join(LOGGING_CONFIG['SAVE_DIR'], model_name)
            
            # Удаляем предыдущую лучшую модель
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                except OSError as e:
                    self.logger.warning(f"Не удалось удалить предыдущую модель: {e}")
            
            # Сохраняем новую лучшую модель
            self.ai.save_model(save_path)
            self.best_model_path = save_path
            
            self.logger.info(f"\nНовая лучшая модель (win rate: {win_rate:.1%}):")
            self.logger.info(f"  {save_path}")
    
    def train(self):
        """Обучение ИИ"""
        self.logger.info("=" * 50)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ")
        self.logger.info("=" * 50)
        
        # Выводим конфигурацию
        self.logger.info("\nКОНФИГУРАЦИЯ:")
        self.logger.info("-" * 20)
        self.logger.info("Игровое поле:")
        self.logger.info(f"  Размер: {GAME_CONFIG['WIDTH']}x{GAME_CONFIG['HEIGHT']}")
        self.logger.info(f"  Мины:   {GAME_CONFIG['NUM_MINES']}")
        
        self.logger.info("\nПараметры обучения:")
        self.logger.info(f"  Эпизоды:        {TRAINING_CONFIG['NUM_EPISODES']}")
        self.logger.info(f"  Batch size:      {TRAINING_CONFIG['BATCH_SIZE']}")
        self.logger.info(f"  Epsilon start:   {TRAINING_CONFIG['INITIAL_EPSILON']}")
        self.logger.info(f"  Epsilon end:     {TRAINING_CONFIG['FINAL_EPSILON']}")
        self.logger.info(f"  Epsilon decay:   {TRAINING_CONFIG['EPSILON_DECAY']}")
        self.logger.info("=" * 50 + "\n")
        
        # Добавляем нормализацию входных данных
        def normalize_state(state):
            return (state - state.mean()) / (state.std() + 1e-8)

        # Добавляем валидацию
        def validate(self, num_games=100):
            wins = 0
            total_reward = 0
            
            # Переключаем модель в режим оценки
            self.ai.model.eval()
            
            with torch.no_grad():
                for _ in range(num_games):
                    game = Minesweeper(
                        GAME_CONFIG['WIDTH'],
                        GAME_CONFIG['HEIGHT'],
                        GAME_CONFIG['NUM_MINES']
                    )
                    reward, won = self.ai.play_episode(game, epsilon=0.05, training=False)
                    wins += int(won)
                    total_reward += reward
            
            # Возвращаем модель в режим обучения
            self.ai.model.train()
            
            return total_reward / num_games, wins / num_games

        epsilon = TRAINING_CONFIG['INITIAL_EPSILON']
        total_rewards = []
        wins = 0
        episode_times = []
        start_time = datetime.now()
        best_win_rate = 0
        patience = 0
        max_patience = 10
        current_win_rate = 0  # Добавляем переменную для отслеживания текущего win rate
        
        for episode in range(TRAINING_CONFIG['NUM_EPISODES']):
            episode_start = datetime.now()
            
            game = Minesweeper(
                GAME_CONFIG['WIDTH'],
                GAME_CONFIG['HEIGHT'],
                GAME_CONFIG['NUM_MINES']
            )
            
            if episode < TRAINING_CONFIG['WARMUP_EPISODES']:
                epsilon = 1.0
            else:
                state = normalize_state(self.ai.get_state(game))
            
            reward, won = self.ai.play_episode(game, epsilon)
            total_rewards.append(reward)
            wins += int(won)
            
            # Вычисляем текущий win rate после каждого эпизода
            if episode > 0:
                current_win_rate = wins / min(episode + 1, LOGGING_CONFIG['LOG_INTERVAL'])
            
            episode_time = (datetime.now() - episode_start).total_seconds()
            episode_times.append(episode_time)
            
            # Уменьшаем epsilon
            epsilon = max(
                TRAINING_CONFIG['FINAL_EPSILON'],
                epsilon * TRAINING_CONFIG['EPSILON_DECAY']
            )
            
            # Сохраняем чекпоинты
            if episode % 1000 == 0:
                checkpoint_path = f"models/checkpoint_episode_{episode}.pth"
                save_checkpoint(self.ai, self.ai.optimizer, episode, current_win_rate, checkpoint_path)
                
                # Проводим валидацию
                val_reward, val_win_rate = self.validate(num_games=100)
                self.logger.info(f"Validation: reward={val_reward:.2f}, win_rate={val_win_rate:.2%}")
            
            # Логируем прогресс
            if (episode + 1) % LOGGING_CONFIG['LOG_INTERVAL'] == 0:
                interval = LOGGING_CONFIG['LOG_INTERVAL']
                avg_reward = sum(total_rewards[-interval:]) / interval
                win_rate = wins / interval
                current_win_rate = win_rate  # Обновляем текущий win rate
                avg_time = sum(episode_times[-interval:]) / interval
                elapsed_time = datetime.now() - start_time
                current_lr = self.ai.get_current_lr()
                
                self.logger.info("-" * 50)
                self.logger.info(f"ЭПИЗОД: {episode + 1}/{TRAINING_CONFIG['NUM_EPISODES']}")
                self.logger.info(f"Прошло времени: {str(elapsed_time).split('.')[0]}")
                self.logger.info(f"Средняя награда: {avg_reward:.3f}")
                self.logger.info(f"Процент побед:   {win_rate:.1%}")
                self.logger.info(f"Epsilon:         {epsilon:.3f}")
                self.logger.info(f"Learning rate:   {current_lr:.6f}")
                self.logger.info(f"Время на эпизод: {avg_time:.3f} сек")
                self.logger.info("-" * 50 + "\n")
                
                wins = 0  # Сбрасываем счетчик побед
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.save_best_model(win_rate)
                    patience = 0
                else:
                    patience += 1
                    
                if patience >= max_patience:
                    self.logger.info("Остановка обучения: нет улучшений")
                    break
        
        # Итоговая статистика
        total_time = datetime.now() - start_time
        final_avg_reward = sum(total_rewards[-100:]) / 100
        
        self.logger.info("=" * 50)
        self.logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self.logger.info("=" * 50)
        self.logger.info(f"\nИтоговая статистика:")
        self.logger.info(f"  Время обучения:    {str(total_time).split('.')[0]}")
        self.logger.info(f"  Средняя награда:   {final_avg_reward:.3f}")
        self.logger.info(f"  Финальный epsilon: {epsilon:.3f}")
        
        # Сохраняем финальную модель
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(
            LOGGING_CONFIG['SAVE_DIR'],
            f'model_final_{timestamp}.pth'
        )
        self.ai.save_model(save_path)
        self.logger.info(f"\nФинальная модель сохранена в:")
        self.logger.info(f"  {save_path}")
        self.logger.info("=" * 50)

    def validate(self, num_games=100):
        """Проверка модели на тестовых играх"""
        wins = 0
        total_reward = 0
        
        # Переключаем модель в режим оценки
        self.ai.model.eval()
        
        with torch.no_grad():
            for _ in range(num_games):
                game = Minesweeper(
                    GAME_CONFIG['WIDTH'],
                    GAME_CONFIG['HEIGHT'],
                    GAME_CONFIG['NUM_MINES']
                )
                reward, won = self.ai.play_episode(game, epsilon=0.05, training=False)
                wins += int(won)
                total_reward += reward
        
        # Возвращаем модель в режим обучения
        self.ai.model.train()
        
        return total_reward / num_games, wins / num_games

    def train_model(self):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LEARNING_RATE_DECAY,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=[early_stopping, lr_reducer]
        )

    def train_batch_parallel(self, batch_indices):
        """Параллельное обучение батча"""
        results = self.pool.map(
            self.train_single_episode,
            [(i, self.preallocated_games[i % len(self.preallocated_games)])
             for i in batch_indices]
        )
        return zip(*results)  # rewards, wins

def save_checkpoint(ai_agent, optimizer, episode, win_rate, path):
    """
    Сохраняет состояние обучения в чекпоинт
    
    Args:
        ai_agent: объект MinesweeperAI
        optimizer: оптимизатор
        episode: текущий эпизод
        win_rate: текущий процент побед
        path: путь для сохранения чекпоинта
    """
    torch.save({
        'episode': episode,
        'model_state_dict': ai_agent.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'win_rate': win_rate
    }, path)

def load_checkpoint(ai_agent, optimizer, path):
    """
    Загружает состояние обучения из чекпоинта
    
    Args:
        ai_agent: объект MinesweeperAI
        optimizer: оптимизатор
        path: путь к чекпоинту
    """
    checkpoint = torch.load(path)
    ai_agent.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['win_rate']

def main():
    trainer = AITrainer()
    trainer.train()

if __name__ == '__main__':
    main() 