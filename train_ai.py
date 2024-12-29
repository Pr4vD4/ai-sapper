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
        
        # Загружаем существующую модель, если это указано в конфиге
        if TRAINING_CONFIG['LOAD_EXISTING_MODEL']:
            self.load_existing_model()
    
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
        
        epsilon = TRAINING_CONFIG['INITIAL_EPSILON']
        total_rewards = []
        wins = 0
        episode_times = []
        start_time = datetime.now()
        
        for episode in range(TRAINING_CONFIG['NUM_EPISODES']):
            episode_start = datetime.now()
            
            game = Minesweeper(
                GAME_CONFIG['WIDTH'],
                GAME_CONFIG['HEIGHT'],
                GAME_CONFIG['NUM_MINES']
            )
            
            reward, won = self.ai.play_episode(game, epsilon)
            total_rewards.append(reward)
            wins += int(won)
            
            episode_time = (datetime.now() - episode_start).total_seconds()
            episode_times.append(episode_time)
            
            # Уменьшаем epsilon
            epsilon = max(
                TRAINING_CONFIG['FINAL_EPSILON'],
                epsilon * TRAINING_CONFIG['EPSILON_DECAY']
            )
            
            # Сохраняем промежуточную модель
            if (LOGGING_CONFIG['SAVE_CHECKPOINTS'] and 
                (episode + 1) % LOGGING_CONFIG['CHECKPOINT_INTERVAL'] == 0):
                checkpoint_path = os.path.join(
                    LOGGING_CONFIG['SAVE_DIR'],
                    f'checkpoint_ep{episode + 1}.pth'
                )
                self.ai.save_model(checkpoint_path)
                self.logger.info(f"\nСохранен чекпоинт модели:")
                self.logger.info(f"  {checkpoint_path}")
            
            # Логируем прогресс
            if (episode + 1) % LOGGING_CONFIG['LOG_INTERVAL'] == 0:
                interval = LOGGING_CONFIG['LOG_INTERVAL']
                # Берем только значения за последний интервал
                avg_reward = sum(total_rewards[-interval:]) / interval
                win_rate = wins / interval
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

def main():
    trainer = AITrainer()
    trainer.train()

if __name__ == '__main__':
    main() 