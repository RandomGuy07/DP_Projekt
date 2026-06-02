import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, lax
from functools import partial    
from jax.example_libraries import optimizers   
import numpy as onp
import pickle

L_x = 2.0
L_y = 0.5
CYLINDER_CENTER_X = 0.5
CYLINDER_CENTER_Y = 0.25
CYLINDER_R = 0.1
NU = 0.002
U_MEAN = 1.0
U_AMPLITUDE = 0.5
INLET_FREQ = 1.0

def MLP(layers, consts_key=random.PRNGKey(12345), rff_n=64, sigma = 2.0, M_t = 2, activation=np.tanh):
    def xavier_init(key, d_in, d_out):
        stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        W = stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    B_key,key = random.split(consts_key)
    B = sigma * random.normal(B_key, (rff_n,2))
    #print(B)
    
    def spatial_encoding(x,y):
        coords = np.array([x,y])
        proj = 2.0 * np.pi * (B @ coords)
        return np.concatenate([np.cos(proj), np.sin(proj)])
    
    def temporal_encoding(t):
        k = np.power(10.0, np.arange(0, M_t +1))
        return k * t

    d_t = M_t + 1
    d_x = 2 * rff_n
    d_in = d_t + d_x
    layers = [d_in] + list(layers[1:]) 


    def init(rng_key):
        key1,key2 = random.split(key)
        U1,b1 = xavier_init(key1, layers[0], layers[1])
        U2,b2 = xavier_init(key2, layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        k, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)


    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        
        t = inputs[0]
        x = inputs[1]
        y = inputs[2]
        
        encoded_time = temporal_encoding(t)
        encoded_space = spatial_encoding(x,y)

        encoded_in = np.concatenate([encoded_time, encoded_space])
        U = activation(np.dot(encoded_in, U1) + b1)
        V = activation(np.dot(encoded_in, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(np.dot(encoded_in, W) + b)
            encoded_in  = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
        W, b = params[-1]
        outputs = np.dot(encoded_in, W) + b
        return outputs

    return init, apply




def inlet_velocity(t):
    return U_MEAN + U_AMPLITUDE * np.sin(2.0 * np.pi * INLET_FREQ *t)

def sample_collocation(key, n_pts, t_0, t_max):
    keys = random.split(key, 4)
    t = random.uniform(keys[0], (2 * n_pts,), minval=t_0, maxval=t_max)
    x = random.uniform(keys[1], (2 * n_pts,), minval=0.0, maxval=L_x)
    y = random.uniform(keys[2], (2 * n_pts,), minval=0.0, maxval=L_y)

    inside_cylinder = (x - CYLINDER_CENTER_X) ** 2 + (y - CYLINDER_CENTER_Y) ** 2 < CYLINDER_R ** 2

    mask = ~inside_cylinder
    idx = np.where(mask, np.arange(2 * n_pts), 2 * n_pts)
    idx = np.sort(idx)[:n_pts]
    return t[idx], x[idx], y[idx]


def sample_boundary(key, pts_boundary, t_0, t_max):
    keys = random.split(key, 9)
    
    #Inlet (x=0)
    t_in = random.uniform(keys[0], (pts_boundary,), minval=t_0, maxval=t_max)
    y_in = random.uniform(keys[1], (pts_boundary,), minval=0.0, maxval=L_y)
    x_in = np.zeros(pts_boundary)

    # Outlet (x=L_x)
    t_out = random.uniform(keys[2], (pts_boundary,), minval=t_0, maxval=t_max)
    y_out = random.uniform(keys[3], (pts_boundary,), minval=0.0, maxval=L_y)
    x_out = L_x * np.ones(pts_boundary)

    # Walls (y=0 or y=L_y)
    t_w = random.uniform(keys[4], (pts_boundary,), minval=t_0, maxval=t_max)
    x_w = random.uniform(keys[5], (pts_boundary,), minval=0.0, maxval=L_x)
    y_w = (random.bernoulli(keys[6], 0.5, (pts_boundary,))).astype(float) * L_y

    # Cylinder surface
    t_c = random.uniform(keys[7], (pts_boundary,), minval=t_0, maxval=t_max)
    theta = random.uniform(keys[8], (pts_boundary,), minval=0.0, maxval=2.0 * np.pi)
    x_c = CYLINDER_CENTER_X + CYLINDER_R * np.cos(theta)
    y_c = CYLINDER_CENTER_Y + CYLINDER_R * np.sin(theta)

    return {
        'inlet':    (t_in,  x_in,  y_in),
        'outlet':   (t_out, x_out, y_out),
        'walls':    (t_w,   x_w,   y_w),
        'cylinder': (t_c,   x_c,   y_c),
    }

class PINN:
    def __init__(self, key, layers, start_data, t_0, t_max, time_caus_n, epsilon=1.0, ic_weight=1e2, bc_weight=1e2):
        self.t_0 = t_0
        self.t_max = t_max
        self.time_caus_n = time_caus_n
        self.epsilon = epsilon
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight

        self.t_caus_chunks = np.linspace(t_0, t_max, time_caus_n)
        self.M = np.triu(np.ones((time_caus_n, time_caus_n)), k=1).T

        self.x_ic, self.y_ic, self.u_ic, self.v_ic, self.p_ic = start_data

        self.init, self.apply = MLP(layers, M_t = 2)
        params = self.init(key)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        self.u_fn = vmap(self.u_net, in_axes=(None, 0, 0, 0))
        self.v_fn = vmap(self.v_net, in_axes=(None, 0, 0, 0))
        self.p_fn = vmap(self.p_net, in_axes=(None, 0, 0, 0))

        self.itercount = iter(range(int(1e9)))
        self.loss_log = []
        self.W_log = []

    def neural_net(self, params, t, x, y):
        out = self.apply(params, np.array([t,x,y]))
        return out[0], out[1], out[2]

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p = self.neural_net(params, t, x, y)
        return p


    def residual(self, params, t, x, y):
        u, v, p = self.neural_net(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)

        # Incompressible NS in velocity-pressure form
        f_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        f_c = u_x + v_y                                 # continuity
        return f_u, f_v, f_c

    def residual_per_t(self, params, t_chunks, x_r, y_r):
        n_t = t_chunks.shape[0]
        n_x = x_r.shape[0]

        t_flat = np.repeat(t_chunks, n_x)
        x_flat = np.tile(x_r, n_t)
        y_flat = np.tile(y_r, n_t)   
        fu, fv, fc = vmap(self.residual, in_axes=(None, 0, 0, 0))(params, t_flat, x_flat, y_flat)

        fu = fu.reshape(n_t, n_x)
        fv = fv.reshape(n_t, n_x)
        fc = fc.reshape(n_t, n_x)
    
        return np.mean(fu**2 + fv**2 + 100.0 * fc**2, axis=1)


    def loss_ic(self, params):
        t_0 = self.t_0 * np.ones_like(self.x_ic)
        u_pred = self.u_fn(params, t_0, self.x_ic, self.y_ic)
        v_pred = self.v_fn(params, t_0, self.x_ic, self.y_ic)
        p_pred = self.p_fn(params, t_0, self.x_ic, self.y_ic)
        return (np.mean((u_pred - self.u_ic) ** 2) +
                np.mean((v_pred - self.v_ic) ** 2) +
                np.mean((p_pred - self.p_ic) ** 2))

    def loss_bc(self, params, bc_batch):
        t_in, x_in, y_in = bc_batch['inlet']
        t_out, x_out, y_out = bc_batch['outlet']
        t_w, x_w, y_w = bc_batch['walls']
        t_c, x_c, y_c = bc_batch['cylinder']

        u_in = self.u_fn(params, t_in, x_in, y_in)
        v_in = self.v_fn(params, t_in, x_in, y_in)
        L_in = (np.mean((u_in - vmap(inlet_velocity)(t_in)) ** 2) + np.mean(v_in ** 2))

        # Outlet: p = 0 (and ∂u/∂x = 0 — enforced via residual implicitly)
        p_out = self.p_fn(params, t_out, x_out, y_out)
        L_out = np.mean(p_out ** 2)

        # Walls: u = v = 0
        u_w = self.u_fn(params, t_w, x_w, y_w)
        v_w = self.v_fn(params, t_w, x_w, y_w)
        L_w = np.mean(u_w ** 2) + np.mean(v_w ** 2)

        # Cylinder: u = v = 0
        u_c = self.u_fn(params, t_c, x_c, y_c)
        v_c = self.v_fn(params, t_c, x_c, y_c)
        L_c = np.mean(u_c ** 2) + np.mean(v_c ** 2)

        return L_in + L_out + L_w + L_c
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        x_r, y_r, bc_batch = batch

        L_ic = self.loss_ic(params)
        L_bc = self.loss_bc(params, bc_batch)
        
        # Per-timestep residual losses
        L_t = self.residual_per_t(params, self.t_caus_chunks, x_r, y_r)
        L_0_ic = 1e2 * L_ic
        L_0_bc = 1e2 * L_bc
        W = lax.stop_gradient(np.exp(-self.epsilon * (self.M @ L_t)))
        L_res = np.mean(W * L_t)

        return self.ic_weight * L_ic + self.bc_weight * L_bc + L_res, W

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        (_, _), g = jax.value_and_grad(self.loss, has_aux=True)(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, key, n_iter=20000, n_x=2048, n_bc=512):
        for it in range(n_iter):
            key, sk_r, sk_bc = random.split(key, 3)
            _, x_r, y_r = sample_collocation(sk_r, n_x, self.t_0, self.t_max)

            bc_batch = sample_boundary(sk_bc, n_bc, self.t_0, self.t_max)
            batch = (x_r, y_r, bc_batch)

            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)

            if it % 500 == 0:
                params = self.get_params(self.opt_state)
                (L_val, W_val) = self.loss(params, batch)
                self.loss_log.append(float(L_val))
                self.W_log.append(float(W_val.min()))
                

def load_cfd_ic(filename):
    data = onp.load(filename, allow_pickle=True).item()
    return (np.array(data['x']), np.array(data['y']),
            np.array(data['u']), np.array(data['v']),
            np.array(data['p']))



if __name__ == '__main__':
 
    LAYERS       = [None, 128, 128, 128, 3]   
    T_0          = 0.0
    T_MAX        = 0.5                      
    TIME_CAUS_N  = 16                       
    EPSILON      = 1.0                      
    N_ITER       = 100000                     
    N_COLLOCATION = 4096                    
    N_BOUNDARY   = 512                     
    SEED         = 1234
    IC_WEIGHT    = 1e3
    BC_WEIGHT    = 1e3



    key = random.PRNGKey(SEED)
    init_key, train_key = random.split(key)
 
    start_data = load_cfd_ic("ic/ic_t0.npy")  
 
    model = PINN(
        init_key,
        layers=LAYERS,
        start_data=start_data,
        t_0=T_0,
        t_max=T_MAX,
        time_caus_n=TIME_CAUS_N,
        epsilon=1e-2,            
        ic_weight=IC_WEIGHT,
        bc_weight=BC_WEIGHT,
    )
 
    epsilon_schedule = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    for i,eps in enumerate(epsilon_schedule):
        model.epsilon = eps
        model.train(train_key, n_iter=N_ITER,
                    n_x=N_COLLOCATION, n_bc=N_BOUNDARY)

 

    onp.save('pinn_log.npy', {
        'loss': model.loss_log,
        'W_min': model.W_log,
    })
 

    params = model.get_params(model.opt_state)
    
    checkpoint = {
        'params': params,
        'config': {
            'layers': LAYERS,
            'rff_n': 64,
            'sigma': 2.0,
            'M_t': 2,
            'consts_key_seed': 12345,
            't_0': T_0,
            't_max': T_MAX,
            'nu': NU,
            'use_causal': True,
            'time_caus_n': TIME_CAUS_N,
            'epsilon_schedule': epsilon_schedule,
            'n_iter_per_eps': N_ITER,
            'n_collocation': N_COLLOCATION,
            'n_boundary': N_BOUNDARY,
            'ic_weight': IC_WEIGHT,
            'bc_weight': BC_WEIGHT,
        },
        'loss_log': model.loss_log,
        'W_log': model.W_log,
        }

    with open('pinn_causal2.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

