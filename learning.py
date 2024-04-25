from blockworld import BlockWorldEnv
import random


class QLearning():
	def __init__(self, env: BlockWorldEnv):
		self.env = env
		self.q_table = {}
		self.learning_rate = 0.1    
		self.discount_factor = 0.99   
		self.exploration_rate = 0.1  

	def train(self, episodes=10000):
		for _ in range(episodes):
			current_state, goal_stat = self.env.reset() 
			done = False

			while not done:
				if random.random() < self.exploration_rate:
					action = random.choice(current_state.get_actions())  # Explore
				else:
					action = self.best_action(current_state, goal_stat)     # Exploit

				(next_state, _) , reward, done = self.env.step(action)
    
				# Bellman equation
				old_value = self.q_table.get((current_state, action, goal_stat), 0)
				next_max = max(self.q_table.get((next_state, a, goal_stat), 0) for a in next_state.get_actions())

				new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
				self.q_table[(current_state, action, goal_stat)] = new_value
    
				current_state = next_state
    
	def best_action(self, state_str, goal_state):
		actions = state_str.get_actions()
		best_act = max(actions, key=lambda x: self.q_table.get((state_str, x, goal_state), 0))
		return best_act

	def act(self, s):
		if random.random() < self.exploration_rate:
			return random.choice(s[0].get_actions())
		return self.best_action(s[0], s[1])

if __name__ == '__main__':
	# Here you can test your algorithm. Stick with N <= 4
	N = 4

	env = BlockWorldEnv(N)
	qlearning = QLearning(env)

	# Train
	qlearning.train()

	# Evaluate
	test_env = BlockWorldEnv(N)

	test_problems = 10
	solved = 0
	avg_steps = []

	for test_id in range(test_problems):
		s = test_env.reset()
		done = False

		print(f"\nProblem {test_id}:")
		print(f"{s[0]} -> {s[1]}")

		for step in range(50): 	# max 50 steps per problem
			a = qlearning.act(s)
			s_, r, done = test_env.step(a)

			print(f"{a}: {s[0]}")

			s = s_

			if done:
				solved += 1
				avg_steps.append(step + 1)
				break

	avg_steps = sum(avg_steps) / len(avg_steps)
	print(f"Solved {solved}/{test_problems} problems, with average number of steps {avg_steps}.")