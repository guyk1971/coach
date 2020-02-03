import os
import sys
sys.path.insert(0,os.path.abspath('..'))

from copy import deepcopy
import tensorflow as tf
# import os

from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters, KNNParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, CsvDataset
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters
from rl_coach.architectures.head_parameters import QHeadParameters
from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.base_parameters import TaskParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace, VectorObservationSpace, StateSpace, RewardSpace

# Get all the outputs of this tutorial out of the 'Resources' folder
os.chdir('Resources')

# the dataset size to collect
DATASET_SIZE = 50000

task_parameters = TaskParameters(experiment_path='.')
####################
# Graph Scheduling #
####################
def set_schedule_params():
    schedule_params = ScheduleParameters()

    # 100 epochs (we run train over all the dataset, every epoch) of training
    schedule_params.improve_steps = TrainingSteps(100)

    # we evaluate the model every epoch
    schedule_params.steps_between_evaluation_periods = TrainingSteps(1)

    # only for when we have an enviroment
    schedule_params.evaluation_steps = EnvironmentEpisodes(10)
    schedule_params.heatup_steps = EnvironmentSteps(DATASET_SIZE)
    return schedule_params

################
#  Environment #
################
env_params = GymVectorEnvironment(level='Acrobot-v1')

def set_agent_params():
    #########
    # Agent #
    #########
    agent_params = DDQNAgentParameters()
    agent_params.network_wrappers['main'].batch_size = 128
    agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)
    agent_params.algorithm.discount = 0.99

    # to jump start the agent's q values, and speed things up, we'll initialize the last Dense layer's bias
    # with a number in the order of the discounted reward of a random policy
    agent_params.network_wrappers['main'].heads_parameters = \
        [QHeadParameters(output_bias_initializer=tf.constant_initializer(-100))]

    # NN configuration
    agent_params.network_wrappers['main'].learning_rate = 0.0001
    agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

    # ER - we'll need an episodic replay buffer for off-policy evaluation
    agent_params.memory = EpisodicExperienceReplayParameters()

    # E-Greedy schedule - there is no exploration in Batch RL. Disabling E-Greedy.
    agent_params.exploration.epsilon_schedule = LinearSchedule(initial_value=0, final_value=0, decay_steps=1)
    agent_params.exploration.evaluation_epsilon = 0
    return agent_params


def train_on_pure_random():
    tf.reset_default_graph()  # just to clean things up; only needed for the tutorial

    schedule_params = set_schedule_params()
    agent_params = set_agent_params()

    graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                        env_params=env_params,
                                        schedule_params=schedule_params,
                                        vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1),
                                        reward_model_num_epochs=30)
    graph_manager.create_graph(task_parameters)
    graph_manager.improve()
    return


def train_using_experience_agent():
    tf.reset_default_graph()  # just to clean things up; only needed for the tutorial

    # Experience Generating Agent parameters
    experience_generating_agent_params = DDQNAgentParameters()
    # schedule parameters
    experience_generating_schedule_params = ScheduleParameters()
    experience_generating_schedule_params.heatup_steps = EnvironmentSteps(1000)
    experience_generating_schedule_params.improve_steps = TrainingSteps(
        DATASET_SIZE - experience_generating_schedule_params.heatup_steps.num_steps)
    experience_generating_schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
    experience_generating_schedule_params.evaluation_steps = EnvironmentEpisodes(1)

    # DQN params
    experience_generating_agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(
        100)
    experience_generating_agent_params.algorithm.discount = 0.99
    experience_generating_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

    # NN configuration
    experience_generating_agent_params.network_wrappers['main'].learning_rate = 0.0001
    experience_generating_agent_params.network_wrappers['main'].batch_size = 128
    experience_generating_agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
    experience_generating_agent_params.network_wrappers['main'].heads_parameters = \
        [QHeadParameters(output_bias_initializer=tf.constant_initializer(-100))]

    # ER size
    experience_generating_agent_params.memory = EpisodicExperienceReplayParameters()
    experience_generating_agent_params.memory.max_size = \
        (MemoryGranularity.Transitions,
         experience_generating_schedule_params.heatup_steps.num_steps +
         experience_generating_schedule_params.improve_steps.num_steps)

    # E-Greedy schedule
    experience_generating_agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, DATASET_SIZE)
    experience_generating_agent_params.exploration.evaluation_epsilon = 0



    schedule_params = set_schedule_params()
    # set the agent params as before
    agent_params = set_agent_params()

    # 50 epochs of training (the entire dataset is used each epoch)
    schedule_params.improve_steps = TrainingSteps(50)

    graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                        experience_generating_agent_params=experience_generating_agent_params,
                                        experience_generating_schedule_params=experience_generating_schedule_params,
                                        env_params=env_params,
                                        schedule_params=schedule_params,
                                        vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1),
                                        reward_model_num_epochs=30,
                                        train_to_eval_ratio=0.5)
    graph_manager.create_graph(task_parameters)
    graph_manager.improve()
    return




def train_on_csv_file(csv_file):
    tf.reset_default_graph()  # just to clean things up; only needed for the tutorial

    schedule_params = set_schedule_params()

    #########
    # Agent #
    #########
    # note that we have moved to BCQ, which will help the training to converge better and faster
    agent_params = DDQNBCQAgentParameters()
    agent_params.network_wrappers['main'].batch_size = 128
    agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)
    agent_params.algorithm.discount = 0.99

    # to jump start the agent's q values, and speed things up, we'll initialize the last Dense layer
    # with something in the order of the discounted reward of a random policy
    agent_params.network_wrappers['main'].heads_parameters = \
        [QHeadParameters(output_bias_initializer=tf.constant_initializer(-100))]

    # NN configuration
    agent_params.network_wrappers['main'].learning_rate = 0.0001
    agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

    # ER - we'll be needing an episodic replay buffer for off-policy evaluation
    agent_params.memory = EpisodicExperienceReplayParameters()

    # E-Greedy schedule - there is no exploration in Batch RL. Disabling E-Greedy.
    agent_params.exploration.epsilon_schedule = LinearSchedule(initial_value=0, final_value=0, decay_steps=1)
    agent_params.exploration.evaluation_epsilon = 0

    # can use either a kNN or a NN based model for predicting which actions not to max over in the bellman equation
    agent_params.algorithm.action_drop_method_parameters = KNNParameters()

    # DATATSET_PATH = 'acrobot_dataset.csv'
    DATATSET_PATH = csv_file
    agent_params.memory = EpisodicExperienceReplayParameters()
    agent_params.memory.load_memory_from_file_path = CsvDataset(DATATSET_PATH, is_episodic=True)

    spaces = SpacesDefinition(state=StateSpace({'observation': VectorObservationSpace(shape=6)}),
                              goal=None,
                              action=DiscreteActionSpace(3),
                              reward=RewardSpace(1))

    graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                        env_params=None,
                                        spaces_definition=spaces,
                                        schedule_params=schedule_params,
                                        vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1),
                                        reward_model_num_epochs=30,
                                        train_to_eval_ratio=0.4)
    graph_manager.create_graph(task_parameters)
    graph_manager.improve()
    return




def main():
    train_on_pure_random()
    # train_using_experience_agent()
    # train_on_csv_file('./acrobot_dataset.csv')



if __name__=='__main__':
    main()
