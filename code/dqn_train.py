"""Example DQN training script (illustrative).
IMPORTANT: This uses a placeholder surrogate model. Replace `surrogate_model.predict()`
with your real ANN predictor or process model for real experiments.
"""
import os
import random
import numpy as np

# Reproducibility (user should also seed TensorFlow before model creation)
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# Action mapping: 0=no-op, 1=temp+10, 2=temp-10, 3=flow+25, 4=flow-25, 5=press+0.1, 6=press-0.1, 7=sarea+5, 8=sarea-5
ACTION_DELTAS = [
    (0.0, 0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0, 0.0),
    (-10.0, 0.0, 0.0, 0.0),
    (0.0, 25.0, 0.0, 0.0),
    (0.0, -25.0, 0.0, 0.0),
    (0.0, 0.0, 0.1, 0.0),
    (0.0, 0.0, -0.1, 0.0),
    (0.0, 0.0, 0.0, 5.0),
    (0.0, 0.0, 0.0, -5.0),
]

STATE_BOUNDS = {
    'temp': (200.0, 280.0),
    'flow': (875.0, 950.0),
    'press': (1.6, 2.0),
    'sarea': (30.0, 50.0)
}

class SimpleSurrogate:
    """Placeholder surrogate: deterministic synthetic function to return a mock H2S removal (%) for demo.
    Replace with: from your_ann_module import ANNModel; ann = ANNModel.load(...); ann.predict(state.reshape(1,-1))
    """
    def predict(self, state):
        # state: [temp, flow, press, sarea]
        t, f, p, s = state
        # synthetic formula (not physical): higher temp & sarea -> better removal; very high flow reduces removal.
        score = 0.6*(t-200)/(280-200) + 0.25*(s-30)/(50-30) - 0.15*(f-875)/(950-875) + 0.1*(p-1.6)/(2.0-1.6)
        score = max(0.0, min(1.0, score))
        return score * 100.0  # percentage

# ---------- Replay buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    def push(self, s, a, r, s2, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buffer)

# ---------- Minimal TF Q-network (created at runtime by user) ----------
def build_q_network(n_actions, seed=SEED):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        raise RuntimeError("TensorFlow is required to build the Q-network. Install tensorflow==2.10.0") from e
    tf.random.set_seed(seed)
    inp = layers.Input(shape=(4,), name='state_input')
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(n_actions, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

def clip_state(state):
    t = min(max(state[0], STATE_BOUNDS['temp'][0]), STATE_BOUNDS['temp'][1])
    f = min(max(state[1], STATE_BOUNDS['flow'][0]), STATE_BOUNDS['flow'][1])
    p = min(max(state[2], STATE_BOUNDS['press'][0]), STATE_BOUNDS['press'][1])
    s = min(max(state[3], STATE_BOUNDS['sarea'][0]), STATE_BOUNDS['sarea'][1])
    return np.array([t, f, p, s], dtype=float)

def train_dqn(episodes=50, steps_per_episode=100, seed=SEED):
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    surrogate = SimpleSurrogate()
    n_actions = len(ACTION_DELTAS)
    online_net = build_q_network(n_actions, seed=seed)
    target_net = build_q_network(n_actions, seed=seed)
    target_net.set_weights(online_net.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    buffer = ReplayBuffer(capacity=10000)
    batch_size = 64
    gamma = 0.99
    target_update_freq = 500
    train_steps = 0

    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.01

    # start from mid-range initial state
    state = np.array([(STATE_BOUNDS['temp'][0]+STATE_BOUNDS['temp'][1])/2.0,
                      (STATE_BOUNDS['flow'][0]+STATE_BOUNDS['flow'][1])/2.0,
                      (STATE_BOUNDS['press'][0]+STATE_BOUNDS['press'][1])/2.0,
                      (STATE_BOUNDS['sarea'][0]+STATE_BOUNDS['sarea'][1])/2.0])

    for ep in range(episodes):
        # optionally randomize start state
        state = np.array([np.random.uniform(*STATE_BOUNDS['temp']),
                          np.random.uniform(*STATE_BOUNDS['flow']),
                          np.random.uniform(*STATE_BOUNDS['press']),
                          np.random.uniform(*STATE_BOUNDS['sarea'])])
        for step in range(steps_per_episode):
            # epsilon-greedy action selection
            if np.random.rand() < eps:
                action = np.random.randint(n_actions)
            else:
                q_vals = online_net.predict(state.reshape(1,-1), verbose=0)[0]
                action = int(np.argmax(q_vals))

            delta = np.array(ACTION_DELTAS[action])
            next_state = clip_state(state + delta)
            h2s_removal = surrogate.predict(next_state)
            param_change = np.sum(np.abs(delta))
            reward = float(h2s_removal - 0.1 * param_change)
            done = False  # episodic termination can be defined by user

            buffer.push(state, action, reward, next_state, done)

            # training step
            if len(buffer) >= batch_size:
                s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                # compute targets
                q_next = target_net.predict(s2_b, verbose=0)
                q_next_max = np.max(q_next, axis=1)
                targets = r_b + (1 - d_b) * gamma * q_next_max
                q_vals = online_net.predict(s_b, verbose=0)
                for i, a_i in enumerate(a_b):
                    q_vals[i, int(a_i)] = targets[i]
                # train
                online_net.train_on_batch(s_b, q_vals)
                train_steps += 1
                if train_steps % target_update_freq == 0:
                    target_net.set_weights(online_net.get_weights())

            state = next_state.copy()

        # decay epsilon
        eps = max(eps_min, eps * eps_decay)
        print(f"Episode {ep+1}/{episodes} finished. eps={eps:.4f}")

    # save model weights for reproducibility
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    online_net.save_weights(os.path.join(save_dir, 'dqn_online_weights.h5'))
    print("Training complete. Weights saved to ./saved_models/dqn_online_weights.h5")


if __name__ == '__main__':
    print('This script is an illustrative example. Replace the surrogate with your trained ANN.\n')
    train_dqn()
