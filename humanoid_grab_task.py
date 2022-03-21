import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy

class humanoid_grab_task():
    def __init__(self):
        self.ndof = 5
        self.l = np.array([0.45, 0.46, 0.49, 0.3, 0.3])
        self.m = np.array([10.0, 15.0, 20.0, 6.0, 4.0])
        self.foot_length = 0.2
        self.T = 6
        self.dt_cost = 0.2
        self.t_cost_array = np.linspace(0, self.T, int(self.T/self.dt_cost)+1)
        self.R = np.eye(len(self.t_cost_array)*self.ndof)
        self.alpha_control = 1e-1
        self.q_0 = None
        self.lw_ = 2
        self.fontsize_ = 24
        self.fig_height = 2

        self.q_min = np.array([0.0, 0.0, -np.pi, -np.pi, 0.0])
        self.q_max = np.array([np.pi, np.pi, 0.0, 0.0, np.pi])
        self.dq_lim = 0.1
        self.ddq_lim = 2.0

        self.x_des = np.array([0.4, 0.1, -np.pi/4.0])

        self.s = np.zeros(self.ndof+1)
        for i in range(1,self.ndof):
            self.s[i] = self.m[i-1] + self.m[i]
        self.s[0] = self.m[0]
        self.s[-1] = self.m[-1]
        self.s *= 0.5

        self.obj_mass = 40.0

    def fk(self, q):
        T = np.tril(np.ones([self.ndof, self.ndof]))
        T2 = np.tril(np.matlib.repmat(self.l, self.ndof, 1))
        f = np.vstack(( 
            T2 @ np.cos(T@q),
            T2 @ np.sin(T@q)
            )).T
        f = np.vstack(( 
            np.zeros(2),
            f
            ))
        return f

    def CoM(self, q):
        return self.s.dot(self.fk(q)).T / np.sum(self.m)
    
    def CoM_T(self, q):
        fk = self.fk(q)
        return (self.s.dot(fk).T + fk[-1] * self.obj_mass) / (np.sum(self.m) + self.obj_mass)

    def set_basis(self, Phi, dPhi, ddPhi):
        self.Phi = Phi
        self.dPhi = dPhi
        self.ddPhi = ddPhi
        self.Rw = self.dt_cost * ddPhi.T @ self.R @ ddPhi

    def control_cost(self, w):
        return 0.5 * w.dot(self.Rw @ w)

    def control_grad(self, w):
        return w @ self.Rw

    def plot_robot(self, ax, x, c):
        link_plots = []
        for i in range(self.ndof):
            link_plots.append(ax.plot([x[i,0], x[i+1,0]], [x[i,1], x[i+1,1]], lw=self.m[i]/2, color=c))
        joint_plots = ax.scatter(x[:,0], x[:,1], s=100, color=c)
        return link_plots, joint_plots

    def plot_environment(self, q):
        fig, ax = plt.subplots(figsize=(10, 5), dpi= 120, facecolor='w', edgecolor='k')
        # foot
        ax.plot([0.0, self.foot_length], [0., 0.], lw=4, color='g')
        # target
        ax.scatter(self.x_des[0], self.x_des[1], c='k', s=200)
        ax.plot([self.x_des[0], self.x_des[0] - self.l[-1]/2 * np.cos(self.x_des[2])], [self.x_des[1], self.x_des[1] - self.l[-1]/2 * np.sin(self.x_des[2])], 'k', lw=6)
        # draw robot links, joints and end effector
        self.plot_robot(ax, self.fk(q), 'g')
        com = self.CoM(q)
        plt.scatter(com[0], com[1], c='g', s=200, marker='x')
        axes = plt.gca()
        axes.set_xlim([-1.6,1.6])
        axes.set_ylim([-0.01,1.6])

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
        fig, ax = plt.subplots(figsize=(12, 6), dpi= 120, facecolor='w', edgecolor='k')

        K = len(q)
        dt = self.T / K
        dt_ani = 1/fps

        # forward kinematics
        Xs = np.zeros((K, self.ndof+1, 2))

        for k in range(K):
            q_ = q[k]
            Xs[k] = self.fk(q_)
            
        # foot
        ax.plot([0.0, self.foot_length], [0., 0.], lw=4, color='g')

        # target
        ax.scatter(self.x_des[0], self.x_des[1], c='k', s=200)
        ax.plot([self.x_des[0], self.x_des[0] - self.l[-1]/2 * np.cos(self.x_des[2])], [self.x_des[1], self.x_des[1] - self.l[-1]/2 * np.sin(self.x_des[2])], 'k', lw=6)

        # draw robot links, joints and end effector
        link_plot, joint_plot = self.plot_robot(ax, Xs[0], 'g')
        com = self.CoM(q[0])
        com_plot = plt.scatter(com[0], com[1], c='g', s=200, marker='x')

        #ax.plot(Xs[:,-1,0], Xs[:,-1,1], lw=1, color='g--')

        axes = plt.gca()
        axes.set_xlim([-1.6,1.6])
        axes.set_ylim([-0.01,1.6])

        def animate(i_):
            i = int(i_*dt_ani/dt)
            for j in range(len(link_plot)):
                link_plot[j][0].set_xdata(Xs[i,j:j+2,0])
                link_plot[j][0].set_ydata(Xs[i,j:j+2,1])
            joint_plot.set_offsets(Xs[i,:-1,:])
            com_plot.set_offsets(self.CoM(q[i]))
            return com_plot,

        ani = animation.FuncAnimation(
        fig, animate, interval=dt_ani*1e3, blit=True, frames=int(self.T*fps), repeat=False)

        gif_name = r"/home/julius/Documents/python/pac/traj_opt/media/" + filename + ".gif" 
        writergif = animation.PillowWriter(fps=30) 
        ani.save(gif_name, writer=writergif)