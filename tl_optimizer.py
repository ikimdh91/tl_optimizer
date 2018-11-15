import numpy as np
import random
from environment import Env
from collections import defaultdict
from xml.etree.ElementTree import ElementTree, parse, dump

import csv

class QLearningAgent:
    def __init__(self, actions):
        # 행동 = [0, 1]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.7
        self.q_table = defaultdict(lambda: [0.0, 0.0])

    def get_state(self,) :
        count = 0
        avg_veh_num = 0

        # Read output csv file (state)
        with open('test-1-node-PeriodicOutput.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                count += 1
                avg_veh_num += float(row['avgvehnum'])
                # avg_speed += float(row['avgspeed'])
                # avg_density += float(row['avgdensity'])

        avg_veh_num = avg_veh_num / count
        # avg_speed = avg_speed / count
        # avg_density = avg_density / count

        return  avg_veh_num

    def get_action(self, state) :
        random_val = np.random.rand()
        #print("random_val ? ==== ", random_val)
        if random_val > self.epsilon :
            #무작위
            action = np.random.choice(self.actions)
        else :
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def run_salt(self, state, action):

        reward = 0
        # call salt simulator
        #call(['./saltalone', config_file])
        next_state = round(agent.get_state(),2)

        # fake_next_state = [1,2,3,4,5]

        if state < next_state :
            reward = -1
        else :
            reward = 1

        return next_state, reward

    def action_policy(self, weight, epoch) :
        # Parse previous traffic light xml file

        previous_tll_file = 'tss.xml'
        tree = parse(previous_tll_file)
        root = tree.getroot()

       # print(root[0][i][0].get("duration"))
        #root[0][5][0], root[0][6][0] ~ root[0][9][0]
        if weight == 0 :
            for i in range(5, 10):
                #print("\nCurrent duration = ", root[0][i][0].get("duration"))
                root[0][i][0].set("duration", str(int(root[0][i][0].get("duration")) + 1))
                #print("Updated duration = ",root[0][i][0].get("duration"))
        else :
            for i in range(5,10) :
                # print("\nCurrent duration = " , root[0][i][0].get("duration"))
                root[0][i][0].set("duration", str(int(root[0][i][0].get("duration")) - 1))
                # print("Updated duration = ", root[0][i][0].get("duration"))

        output_tll_file = 'tll_epoch-' + str(epoch) + '.xml'
        tree.write(output_tll_file)

    def learn(self, state, action, reward, next_state, epoch):
        q_1 = self.q_table[state][action]

        # 벨만 최적 방정식을 사용한 큐함수 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

        if reward == 1 :
            agent.action_policy(0, epoch)
        else :
            agent.action_policy(1, epoch)


        # print("\n---------------------------Q-TABLE---------------------------\n",self.q_table.items())



if __name__ == "__main__":

    # n_actions=['0','1','2','3','4']
    agent = QLearningAgent(actions=[0,1])

    state = round(agent.get_state(),2)
    # print(state)

    epoch = 0
    total_reward = 0
    for i in range(0,5) :
        epoch+=1

        state = round(agent.get_state(), 2)
        # print(state)

        #현재 상태에 대한 행동 선택
        action = agent.get_action(str(state))
        # print("\n-------------------------------------------ACTION = ", action)

        # 행동을 취한 후 다음 상태, 보상, 등을 받아옴
        next_state, reward = agent.run_salt(state, action)

        total_reward+=reward
        #<s,a,r,s'>로 큐함수를 업데이트
        agent.learn(str(state), action, reward, str(next_state), epoch)

        print("\nCURRENT STATE : ", state, "        NEXT_STATE = ",next_state)

        state = next_state
    # print("TOTAL REWARD : ",total_reward)







