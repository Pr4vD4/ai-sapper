import unittest
import torch
from minesweeper_ai import MinesweeperAI, MinesweeperNet
from main import Minesweeper
import numpy as np
from logger_config import setup_logger

class TestMinesweeperAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Настройка перед всеми тестами"""
        cls.logger = setup_logger('minesweeper_tests')
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.width = 3
        self.height = 3
        self.num_mines = 2
        self.game = Minesweeper(self.width, self.height, self.num_mines)
        self.ai = MinesweeperAI(self.width, self.height, self.num_mines)
        self.logger.info(f"\nЗапуск теста: {self._testMethodName}")

    def tearDown(self):
        """Действия после каждого теста"""
        self.logger.info(f"Завершение теста: {self._testMethodName}\n")

    def test_get_state(self):
        """Проверяем корректность преобразования состояния игры"""
        state = self.ai.get_state(self.game)
        
        # Проверяем размерность состояния
        # Для каждой клетки у нас 2 параметра (открыта/закрыта и количество мин рядом)
        expected_size = self.width * self.height * 2
        self.assertEqual(len(state), expected_size)
        
        # Проверяем, что все значения находятся в допустимом диапазоне
        self.assertTrue(all(0 <= x <= 1 for x in state))

    def test_choose_action(self):
        """Проверяем, что выбор действия возвращает допустимые координаты"""
        x, y = self.ai.choose_action(self.game)
        
        # Проверяем, что координаты находятся в пределах поля
        self.assertTrue(0 <= x < self.width)
        self.assertTrue(0 <= y < self.height)
        
        # Проверяем, что выбранная клетка не открыта
        self.assertFalse(self.game.board[y][x].is_revealed)

    def test_neural_network_output(self):
        """Проверяем, что нейронная сеть выдает корректный формат выхода"""
        state = self.ai.get_state(self.game)
        # Переносим тензор на то же устройство, что и модель
        state_tensor = torch.FloatTensor(state).to(self.ai.device)
        
        output = self.ai.model.network(state_tensor)
        
        self.assertEqual(output.shape, torch.Size([1]))
        # Переносим результат на CPU для сравнения
        output_value = output.cpu().item()
        self.assertTrue(0 <= output_value <= 1)

    def test_reward_calculation(self):
        """Проверяем правильность вычисления наград"""
        self.logger.info("Тестирование вычисления наград")
        
        game = Minesweeper(3, 3, 1)
        ai = self.ai
        
        # Тест награды за безопасный ход
        prev_revealed = 0
        x, y = 0, 0
        while game.board[y][x].is_mine:
            x = (x + 1) % 3
        
        self.logger.debug(f"Тестирование безопасного хода в позиции ({x}, {y})")
        game.reveal(x, y)
        reward = ai.calculate_reward(game, (x, y), prev_revealed)
        self.logger.info(f"Награда за безопасный ход: {reward}")
        self.assertGreater(reward, 0)
        
        # Тест награды за проигрыш
        mine_x, mine_y = None, None
        for y in range(3):
            for x in range(3):
                if game.board[y][x].is_mine:
                    mine_x, mine_y = x, y
                    break
        
        self.logger.debug(f"Тестирование хода на мину в позиции ({mine_x}, {mine_y})")
        game.reveal(mine_x, mine_y)
        reward = ai.calculate_reward(game, (mine_x, mine_y), prev_revealed)
        self.logger.info(f"Награда за проигрыш: {reward}")
        self.assertEqual(reward, -1.0)

    def test_state_representation(self):
        """Проверяем корректность представления состояния игры"""
        self.logger.info("Тестирование представления состояния")
        
        game = Minesweeper(3, 3, 1)
        state = self.ai.get_state(game)
        
        self.logger.debug(f"Начальное состояние: {state}")
        self.logger.debug(f"Размерность состояния: {len(state)}")
        
        # Открываем клетку и проверяем изменение состояния
        x, y = 0, 0
        while game.board[y][x].is_mine:
            x = (x + 1) % 3
        
        self.logger.debug(f"Открываем клетку ({x}, {y})")
        game.reveal(x, y)
        new_state = self.ai.get_state(game)
        self.logger.debug(f"Новое состояние: {new_state}")
        
        # Проверяем изменения
        changes = np.where(state != new_state)[0]
        self.logger.info(f"Изменившиеся индексы: {changes}")
        self.assertFalse(np.array_equal(state, new_state))

    def test_action_selection(self):
        """Проверяем выбор действий"""
        self.logger.info("Тестирование выбора действий")
        
        game = Minesweeper(3, 3, 1)
        
        # Тест случайного выбора
        self.logger.debug("Тестирование случайного выбора (epsilon=1.0)")
        action = self.ai.choose_action(game, epsilon=1.0)
        self.logger.info(f"Случайное действие: {action}")
        
        # Тест выбора через нейросеть
        self.logger.debug("Тестирование выбора через нейросеть (epsilon=0.0)")
        action = self.ai.choose_action(game, epsilon=0.0)
        self.logger.info(f"Действие нейросети: {action}")
        
        x, y = action
        self.logger.debug(f"Проверка допустимости действия ({x}, {y})")
        self.assertTrue(0 <= x < 3 and 0 <= y < 3)
        self.assertFalse(game.board[y][x].is_revealed)

    def test_action_diversity(self):
        """Проверка разнообразия выбираемых действий"""
        self.logger.info("Тестирование разнообразия действий")
        
        game = Minesweeper(3, 3, 1)
        actions = set()
        num_trials = 100
        
        self.logger.debug(f"Сбор {num_trials} действий с epsilon=0.3")
        for _ in range(num_trials):
            action = self.ai.choose_action(game, epsilon=0.3)
            actions.add(action)
        
        self.logger.info(f"Уникальные действия: {actions}")
        self.logger.info(f"Количество уникальных действий: {len(actions)}")
        
        min_expected_actions = 3
        self.assertGreaterEqual(
            len(actions), 
            min_expected_actions,
            f"ИИ должен выбирать как минимум {min_expected_actions} разных действий"
        )

    def test_reward_sequence(self):
        """Проверка последовательности наград"""
        self.logger.info("Тестирование последовательности наград")
        
        game = Minesweeper(3, 3, 1)
        rewards = []
        actions_taken = []
        
        # Играем одну полную игру
        while not game.game_over:
            prev_revealed = sum(1 for row in game.board 
                              for cell in row if cell.is_revealed)
            
            action = self.ai.choose_action(game, epsilon=0)
            actions_taken.append(action)
            
            game.reveal(action[0], action[1])
            reward = self.ai.calculate_reward(game, action, prev_revealed)
            rewards.append(reward)
            
            self.logger.debug(f"Ход {len(actions_taken)}: {action}, Награда: {reward}")
        
        self.logger.info(f"Последовательность действий: {actions_taken}")
        self.logger.info(f"Последовательность наград: {rewards}")
        self.logger.info(f"Итоговая награда: {sum(rewards)}")
        
        # Проверяем, что награды имеют смысл
        self.assertTrue(all(r >= -1.0 and r <= 1.0 for r in rewards), 
                       "Все награды должны быть в диапазоне [-1, 1]")

    def test_safe_move_reward_calculation(self):
        """Детальная проверка вычисления наград за безопасные ходы"""
        self.logger.info("Тестирование вычисления наград за безопасные ходы")
        
        game = Minesweeper(3, 3, 1)
        
        # Находим безопасную клетку
        safe_x, safe_y = 0, 0
        while game.board[safe_y][safe_x].is_mine:
            safe_x = (safe_x + 1) % 3
        
        # Проверяем награду за первый ход
        prev_revealed = 0
        game.reveal(safe_x, safe_y)
        reward = self.ai.calculate_reward(game, (safe_x, safe_y), prev_revealed)
        
        current_revealed = sum(1 for row in game.board 
                             for cell in row if cell.is_revealed)
        cells_opened = current_revealed - prev_revealed
        
        self.logger.info(f"Открыто клеток: {cells_opened}")
        self.logger.info(f"Награда: {reward}")
        self.logger.info(f"Награда на клетку: {reward/cells_opened if cells_opened else 0}")
        
        # Обновляем ожидаемую награду с учетом дополнительной награды за подсказки
        base_reward = 0.1 * cells_opened
        hint_reward = 0.05 * game.board[safe_y][safe_x].adjacent_mines if game.board[safe_y][safe_x].adjacent_mines > 0 else 0
        expected_reward = base_reward + hint_reward
        
        self.assertAlmostEqual(
            reward, 
            expected_reward, 
            places=5,
            msg=f"Награда {reward} не соответствует ожидаемой {expected_reward}"
        )

    def test_learning_progress(self):
        """Проверка прогресса обучения"""
        self.logger.info("Тестирование прогресса обучения")
        
        # Сохраняем начальные веса
        initial_weights = {
            name: param.clone().detach()
            for name, param in self.ai.model.named_parameters()
        }
        
        # Увеличиваем количество эпизодов
        num_episodes = 20
        epsilon = 0.5  # Увеличиваем исследование
        
        self.logger.debug("Начало мини-обучения")
        total_rewards = []
        
        for episode in range(num_episodes):
            game = Minesweeper(3, 3, 1)
            try:
                # Принудительно обучаем на каждом шаге
                reward, won = self.ai.play_episode(game, epsilon=epsilon)
                total_rewards.append(reward)
                self.logger.debug(
                    f"Эпизод {episode + 1}: "
                    f"награда = {reward:.3f}, "
                    f"победа = {won}, "
                    f"loss = {self.ai.batch_losses[-1] if self.ai.batch_losses else 'N/A'}"
                )
            except ValueError as e:
                self.logger.warning(f"Эпизод {episode + 1} прерван: {str(e)}")
                continue
        
        # Проверяем изменение весов
        weights_changed = False
        total_diff = 0.0
        
        for name, param in self.ai.model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                diff = torch.abs(param - initial_weights[name]).mean().item()
                total_diff += diff
                self.logger.info(f"Изменение весов {name}: {diff:.6f}")
        
        self.logger.info(f"Общее изменение весов: {total_diff:.6f}")
        self.logger.info(f"Средняя награда: {sum(total_rewards) / len(total_rewards):.3f}")
        
        # Проверяем, что веса действительно изменились
        self.assertTrue(
            weights_changed and total_diff > 1e-4,  # Уменьшаем порог
            f"Веса модели должны значительно измениться в процессе обучения (изменение: {total_diff})"
        )

if __name__ == '__main__':
    unittest.main() 