import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

class Human():
    def __init__(self, x0, dir_, speed, dt):
        self.x = x0         # [x,y] position
        self.dir = dir_     # moving direction from 0-15
        self.speed = speed
        self.dt = dt
        self.trajectory = [self.x.copy()]

    def reset(self):
        self.trajectory = [self.x.copy()]

    def step(self):
        self.x += self.dt * self.speed * np.array([np.cos(self.dir*np.pi/8), np.sin(self.dir*np.pi/8)])
        self.trajectory.append(self.x.copy())

    def predict(self, t):
        x_pred = np.zeros((len(t), 2))
        dx = self.speed * np.array([np.cos(self.dir*np.pi/8), np.sin(self.dir*np.pi/8)])
        for k in range(len(t)):
            x_pred[k] = self.x + (t[k] - t[0]) * dx
        return x_pred

class mobile_navigation_task():
    def __init__(self, T_lookahead=None):
        self.ndof = 2
        self.x_obs = []
        self.x_dyn_obs = []
        self.r_obs = 0.5
        self.r_obs_sq = self.r_obs**2
        self.T = None
        self.dt_planning = 0.04
        self.dt_cost = 0.04
        self.alpha_control = 1e-1
        self.q_0 = None
        self.dq_0 = None
        self.q_d = None
        self.dq_lim = 1.0
        self.lw_ = 2
        self.fontsize_ = 24
        self.fig_height = 2

        self.T_lookahead = T_lookahead

    def set_time(self, T):
        self.T = np.round(T,2)
        self.t_plan_array = np.linspace(0, self.T, int(self.T/self.dt_planning)+1)
        self.t_cost_array = np.linspace(0, self.T, int(self.T/self.dt_cost)+1)
        self.R = np.eye(len(self.t_cost_array)*self.ndof)
        if self.T_lookahead is None:
            self.T_lookahead = self.T

    def set_obstacle_list(self, x_obs):
        self.x_obs = x_obs

    def set_dyn_obstacle_list(self, x_obs):
        self.x_dyn_obs = x_obs

    def get_dist_vec_(self, q_traj, s):
        T_lookahead = np.min([self.T_lookahead, self.T])
        k_end = int(np.round(T_lookahead/self.dt_cost))-2
        e = (q_traj - s)[:k_end]
        return np.sum(e * e, axis=1)

    def get_dist_vec(self, w):
        if len(self.t_cost_array) < 3:
            return self.r_obs_sq
        T_lookahead = np.min([self.T_lookahead, self.T])
        k_end = int(np.round(T_lookahead/self.dt_cost))-2
        if k_end < 1:
            return self.r_obs_sq
        q = (self.Phi[self.ndof:-self.ndof] @ w).reshape(len(self.t_cost_array)-2, self.ndof)
        d = self.r_obs_sq * np.ones_like(self.t_cost_array[1:k_end+1])
        for s in self.x_obs:
            d = np.minimum(d, self.get_dist_vec_(q, s))
        for s in self.x_dyn_obs:
            d = np.minimum(d, self.get_dist_vec_(q, s))
        return d

    def set_basis(self, Phi, dPhi, ddPhi):
        self.Phi = Phi
        self.dPhi = dPhi
        self.ddPhi = ddPhi
        self.Rw = self.dt_cost * ddPhi.T @ self.R @ ddPhi

    def control_cost(self, w):
        return 0.5 * w.dot(self.Rw @ w)

    def control_grad(self, w):
        return w @ self.Rw

    def plot_environment(self, q, q_via=None):
        plt.figure(figsize=(8, 8), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$q_1$', fontsize=self.fontsize_)
        plt.ylabel(r'$q_2$', fontsize=self.fontsize_)
        if q is not None:
            if len(q.shape) == 2:
                plt.plot(q[:,0], q[:,1], lw=self.lw_)
            else:
                plt.plot(q[:,:,0].T, q[:,:,1].T, lw=1)
        if q_via is not None:
            plt.scatter(q_via[:,0], q_via[:,1])
        plt.scatter(self.q_0[0], self.q_0[1], s=100, color='b')
        plt.scatter(self.q_d[0], self.q_d[1], s=100, color='g')

        ax = plt.gca()
        ax.set_xlim([-1,8])
        ax.set_ylim([-1,8])
        obs_list = []
        for s in self.x_obs:
            obs_list.append(plt.Circle((s[0], s[1]), self.r_obs, color='r'))
            ax.add_patch(obs_list[-1])

        plt.tight_layout()

    def plot_q(self, q):
        T = len(q)*self.dt_planning
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$q$', fontsize=self.fontsize_)
        plt.plot(np.linspace(0, T, len(q)), q, lw=self.lw_)
        plt.plot([0, T], self.q_d[0]*np.ones(2), 'k--')
        plt.plot([0, T], self.q_d[1]*np.ones(2), 'k--')
        plt.tight_layout()

    def plot_dq(self, dq):
        T = len(dq)*self.dt_planning
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\dot{q}$', fontsize=self.fontsize_)
        plt.plot(np.linspace(0, T, len(dq)), dq, lw=self.lw_)
        plt.tight_layout()

    def plot_ddq(self, ddq):
        T = len(ddq)*self.dt_planning
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\ddot{q}$', fontsize=self.fontsize_)
        plt.plot(np.linspace(0, T, len(ddq)), ddq, lw=self.lw_)
        plt.tight_layout()

    def save_animation(self, q, filename="animation", fps=25):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

        K = len(q)
        dt_ani = 1./fps

        for s in self.x_obs:
            ax.add_patch(plt.Circle((s[0], s[1]), self.r_obs/2, color='r'))

        obs_list = []
        for s in self.x_dyn_obs:
            obs_list.append(plt.Circle((s[0,0], s[0,1]), self.r_obs/2, color='r'))
            ax.add_patch(obs_list[-1])

        robot_patch = plt.Circle((q[0,0], q[0,1]), self.r_obs/2, color='k')
        ax.add_patch(robot_patch)

        plt.scatter(self.q_0[0], self.q_0[1], s=100, color='b')
        plt.scatter(self.q_d[0], self.q_d[1], s=100, color='g')

        axes = plt.gca()
        axes.set_xlim([-10,10])
        axes.set_ylim([-10,10])

        def animate(i_):
            i = int(i_*dt_ani/self.dt_planning)
            robot_patch.center = (q[i,0], q[i,1])
            for j in range(len(obs_list)):
                obs_list[j].center = (self.x_dyn_obs[j][i,0], self.x_dyn_obs[j][i,1])
            return robot_patch,

        ani = animation.FuncAnimation(
        fig, animate, interval=dt_ani*1e3, blit=True, frames=int(self.T*fps), repeat=False)

        video_name = r"/home/julius/Documents/python/pac/traj_opt/media/" + filename + ".mp4" 
        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save(video_name, writer=writervideo)