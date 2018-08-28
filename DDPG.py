import tensorflow as tf
import numpy as np
import gym


class DDPG:
    def __init__(self, lr_a, lr_c, dim_features, dim_actions, gamma, tau, memory_capacity, batch_size, bound_actions):
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.dim_features = dim_features
        self.dim_actions = dim_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.bound_actions = bound_actions

        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, self.dim_features * 2 + dim_actions + 1), dtype=np.float32)
        self.memory_pointer = 0

        self.init_w = tf.random_normal_initializer(0., 0.1)
        self.init_b = tf.random_normal_initializer(0., 0.01)

        # input
        self.s = tf.placeholder(tf.float32, [None, self.dim_features], "state_in")
        self.r = tf.placeholder(tf.float32, [None, 1], "reward_in")
        self.s_ = tf.placeholder(tf.float32, [None, self.dim_features], "state_next_in")
        self.a = tf.placeholder(tf.float32, [None, self.dim_actions], "action_in")
        # other interface
        # self.a_chosen = tf.placeholder(tf.float32, [None, dim_actions], "action_chosen")
        # self.pd = tf.placeholder(tf.float32, [None, dim_actions], "partial_derivative")

        with tf.variable_scope("main_net"):
            self.a_main = self.build_a(self.s)
            self.q_main = self.build_c(tf.concat([self.s, self.a], 1))
            self.op_gradient_q_a = tf.gradients(self.q_main, self.a, name="gradient_q_a")
            self.gradient_q_a = tf.placeholder(tf.float32, [None, dim_actions])

        with tf.variable_scope("target_net"):
            a_target = self.build_a(self.s_, False)
            self.q_target = self.build_c(tf.concat([self.s_, a_target], 1), False)
            self.q_real = self.r + self.gamma * self.q_target
            self.q_estimated = self.q_main

        self.param_a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_net/actor")
        self.param_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_net/critic")
        self.param_a_ = tf.get_collection(tf.GraphKeys.VARIABLES, scope="target_net/actor")
        self.param_c_ = tf.get_collection(tf.GraphKeys.VARIABLES, scope="target_net/critic")

        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea),
                              tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.param_a_, self.param_a, self.param_c_, self.param_c)]
        self.hard_replace = [[tf.assign(ta, ea),
                              tf.assign(tc, ec)]
                             for ta, ea, tc, ec in zip(self.param_a_, self.param_a, self.param_c_, self.param_c)]

        with tf.variable_scope("loss"):
            self.J = tf.losses.compute_weighted_loss(self.a_main, self.gradient_q_a, scope="J")
            self.loss_q = tf.losses.mean_squared_error(labels=self.q_real, predictions=self.q_estimated, scope="loss_q")

        self.train_op_a = tf.train.RMSPropOptimizer(learning_rate=self.lr_a).minimize(-self.J, var_list=self.param_a)
        self.train_op_c = tf.train.RMSPropOptimizer(learning_rate=self.lr_c).minimize(self.loss_q, var_list=self.param_c)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.hard_replace)

    def choose_action(self, s, with_noise=True, var=0):
        a = self.sess.run(self.a_main, feed_dict={self.s: s})
        if with_noise:
            return np.clip(np.random.normal(a, var), -2, 2)
        else:
            return a

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]

        bs = bt[:, :self.dim_features]
        ba = bt[:, self.dim_features: self.dim_features + self.dim_actions]
        br = bt[:, -self.dim_features - 1: -self.dim_features]
        bs_ = bt[:, -self.dim_features:]

        q_real = self.sess.run(self.q_real, feed_dict={self.s_: bs_, self.r: br})
        self.sess.run(self.train_op_c, feed_dict={self.q_real: q_real, self.s: bs, self.a: ba, self.r: br})

        a_chosen = self.sess.run(self.a_main, feed_dict={self.s: bs})
        gradient_q_a = self.sess.run(self.op_gradient_q_a, feed_dict={self.s: bs, self.a: a_chosen})
        self.sess.run(self.train_op_a, feed_dict={self.s: bs, self.gradient_q_a: gradient_q_a[0]})

        self.sess.run(self.soft_replace)

    def build_a(self, tensor_in, trainable=True):
        with tf.variable_scope("actor"):
            l1 = tf.layers.dense(inputs=tensor_in, units=30,
                                 activation=tf.nn.elu,
                                 kernel_initializer=self.init_w,
                                 bias_initializer=self.init_b,
                                 name="l1",
                                 trainable=trainable)
            action = tf.layers.dense(inputs=l1, units=self.dim_actions,
                                     activation=tf.nn.tanh,
                                     kernel_initializer=self.init_w,
                                     bias_initializer=self.init_b,
                                     name="l2",
                                     trainable=trainable)
            result = tf.multiply(action, self.bound_actions)
        return result

    def build_c(self, tensor_in, trainable=True):
        with tf.variable_scope("critic"):
            l1 = tf.layers.dense(inputs=tensor_in, units=30,
                                 activation=tf.nn.elu,
                                 kernel_initializer=self.init_w,
                                 bias_initializer=self.init_b,
                                 name="l1",
                                 trainable=trainable)
            q = tf.layers.dense(inputs=l1, units=1,
                                activation=None,
                                kernel_initializer=self.init_w,
                                bias_initializer=self.init_b,
                                name="q",
                                trainable=trainable)
            result = tf.multiply(q, self.bound_actions)
        return result

    def store_memory(self, s, a, r, s_):
        # print("{}\n{}\n{}\n{}\n".format(s, a, r, s_))
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_pointer += 1


MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002   # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 128

RENDER = False
ENV_NAME = 'Pendulum-v0'

INNER_ROUTE = [1]
OUTER_ROUTE = [0]
VAR = 3


if __name__ == "__main__":
    # build environment
    print("/****************************welcome**************************/")
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    # print(a_bound)
    # exit()
    ddpg = DDPG(lr_a=LR_A,
                lr_c=LR_C,
                dim_features=s_dim,
                dim_actions=a_dim,
                gamma=GAMMA,
                tau=TAU,
                bound_actions=a_bound,
                memory_capacity=MEMORY_CAPACITY,
                batch_size=BATCH_SIZE)

    observation = env.reset()
    observation = observation[np.newaxis, :]
    # print("--ob--{}".format(len(observation)))
    for i in range(MEMORY_CAPACITY):
        forward_observation = observation
        action_chosen = ddpg.choose_action(observation, var=VAR)
        observation, reward, done, info = env.step(action_chosen)
        observation = np.array(observation).T
        ddpg.store_memory(forward_observation, action_chosen, reward, observation)
        ddpg.learn()

    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        observation = observation[np.newaxis, :]
        print(i_episode)
        for t in range(MAX_EP_STEPS):
            VAR *= 0.9995
            if i_episode >= 10:
                env.render()
            # action = env.action_space.sample()
            forward_observation = observation
            action_chosen = ddpg.choose_action(observation, var=VAR)
            observation, reward, done, info = env.step(action_chosen)
            # print(action_chosen)
            observation = np.array(observation).T
            ddpg.store_memory(forward_observation, action_chosen, reward, observation)
            ddpg.learn()
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

