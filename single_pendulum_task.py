import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class single_pendulum_task():
    def __init__(self):
        self.ndof = 1
        self.l = 1.0 # np.array([0.45, 0.46, 0.49, 0.3, 0.3])
        self.m = 10.0 # np.array([10.0, 15.0, 20.0, 6.0, 4.0])
        self.g = 9.81
        self.tau_lim = 80.0
        self.T = 6
        self.dt_cost = 0.02
        self.t_cost_array = np.linspace(0, self.T, int(self.T/self.dt_cost)+1)
        self.R = self.m * np.eye(len(self.t_cost_array)*self.ndof)
        self.alpha_target = 1e2
        self.alpha_control = 1e-4
        self.q_0 = np.array([0.0])
        self.q_d = np.array([np.pi])
        self.lw_ = 2
        self.fontsize_ = 24
        self.fig_height = 2

    def inverse_dynamics(self, w):
        return self.m*self.l**2 * self.ddPhi @ w + self.m*self.g*self.l*np.sin(self.Phi @ w)

    def plot_robot(self, ax, q, c):
        ax.plot([0.0, self.l*np.sin(q[0])], [0.0, -self.l*np.cos(q[0])], lw=1, color=c)
        ax.scatter(self.l*np.sin(q[0]), -self.l*np.cos(q[0]), s=400, color=c)

    def set_basis(self, Phi, dPhi, ddPhi):
        self.Phi = Phi
        self.dPhi = dPhi
        self.ddPhi = ddPhi
        self.Rw = self.dt_cost * ddPhi.T @ self.R @ ddPhi

    def target_cost(self, w):
        q = self.Phi[-self.ndof:] @ w
        return 0.5 * (q - np.pi)**2

    def target_grad(self, w):
        q = self.Phi[-self.ndof:] @ w
        return (q - np.pi) * self.Phi[-self.ndof:]

    def control_cost(self, w):
        return 0.5 * w.dot(self.Rw @ w)

    def control_grad(self, w):
        return w @ self.Rw

    def plot_q(self, w):
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$q$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, self.Phi @ w, lw=self.lw_)
        plt.plot(self.t_cost_array, np.pi*np.ones_like(self.t_cost_array), 'k--')
        plt.tight_layout()

    def plot_dq(self, w):
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\dot{q}$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, self.dPhi @ w, lw=self.lw_)
        plt.tight_layout()

    def plot_ddq(self, w):
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\ddot{q}$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, self.ddPhi @ w, lw=self.lw_)
        plt.tight_layout()

    def plot_tau(self, w):
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\tau$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, self.inverse_dynamics(w), lw=self.lw_)
        plt.plot([0,self.T], [self.tau_lim, self.tau_lim], 'k--')
        plt.plot([0,self.T], [-self.tau_lim, -self.tau_lim], 'k--')
        plt.tight_layout()

    def save_animation(self, w, filename="animation"):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120, facecolor='w', edgecolor='k')
        q = (self.Phi @ w).reshape(len(self.t_cost_array), self.ndof)
        K = len(q)
        dt = self.T / K
        dt_ani = 0.04

        # forward kinematics
        Xs = np.zeros((K, self.ndof+1, 2))

        for k in range(K):
            q_ = q[k]
            Xs[k] = np.array([[0.0, 0.0], [self.l*np.sin(q_[0]), -self.l*np.cos(q_[0])]])
        self.plot_robot(ax, [np.pi], 'k')

        # draw robot links, joints and end effector
        link_plot, = ax.plot(Xs[0,:,0], Xs[0,:,1], lw=1, color='g')
        joint_plot = ax.scatter(Xs[0,1:,0], Xs[0,1:,1], s=300, color='g')

        ax.plot(Xs[:,-1,0], Xs[:,-1,1], 'g--')

        axes = plt.gca()
        axes.set_xlim([-1.6,1.6])
        axes.set_ylim([-1.6,1.6])

        def animate(i_):
            i = int(i_*dt_ani/dt)
            link_plot.set_xdata(Xs[i,:,0])
            link_plot.set_ydata(Xs[i,:,1])
            joint_plot.set_offsets(Xs[i,1:,:])
            return link_plot, joint_plot

        ani = animation.FuncAnimation(
        fig, animate, interval=dt_ani*1e3, blit=True, frames=int(self.T/dt_ani), repeat=False)

        gif_name = r"/home/julius/Documents/python/pac/traj_opt/media/" + filename + ".gif" 
        writergif = animation.PillowWriter(fps=30) 
        ani.save(gif_name, writer=writergif)