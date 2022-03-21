import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

class double_pendulum_task():
    def __init__(self):
        self.ndof = 2
        self.l = np.array([0.8, 0.8])
        body_mass = 70.0
        converted_link_mass = 2 * body_mass / 3.
        self.m = converted_link_mass * np.ones(2)
        self.g = 9.81
        self.q_min = np.array([-2.0*np.pi, -np.pi/6.0])
        self.q_max = np.array([2.0*np.pi, 5.0*np.pi/6.0])
        self.tau_lim = np.array([40.0, 500.0])
        self.T = 6
        self.dt_cost = 0.05
        self.t_cost_array = np.linspace(0, self.T, int(self.T/self.dt_cost)+1)
        self.R = np.eye(len(self.t_cost_array)*self.ndof)
        for i in range(len(self.t_cost_array)):
            self.R[i*self.ndof:(i+1)*self.ndof, i*self.ndof:(i+1)*self.ndof] = np.diag(self.m)
        self.alpha_target = 1e2
        self.alpha_control = 1e-4
        self.q_0 = np.array([0.0, 0.0]) # np.array([-np.pi/2., np.pi/2.])
        self.q_d = np.array([np.pi, 0.0])
        self.lw_ = 2
        self.fontsize_ = 24
        self.fig_height = 2

    def M_(self, q):
        T = np.size(q[0])
        c1 = np.cos(q[1])
        m00 = 0.25*self.m[0]*self.l[0]**2 + self.m[1]*(self.l[0]**2 + 0.25*self.l[1]**2 + self.l[0]*self.l[1]*c1)
        m01 = 0.5*self.m[1]*(0.5*self.l[1]**2 + self.l[0]*self.l[1]*c1)
        m11 = self.m[1] * self.l[1]**2 / 4. * np.ones_like(c1)
        M = np.zeros((T, self.ndof, self.ndof))
        M[:, 0, 0] = m00
        M[:, 0, 1] = m01
        M[:, 1, 0] = m01
        M[:, 1, 1] = m11
        return scipy.linalg.block_diag(*M)

    def M(self, w):
        return self.M_((self.Phi @ w).reshape(len(self.t_cost_array), self.ndof).T)

    def b_(self, q, dq):
        T = np.size(q[0])
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s01 = np.sin(q[0] + q[1])
        b0 = 0.5*self.m[1]*self.g*self.l[1]*s01 + (0.5*self.m[0]+self.m[1])*self.g*self.l[0]*s0 - self.m[1]*self.l[0]*self.l[1]*s1*dq[1]*(dq[0]+0.5*dq[1])
        b1 = 0.5*self.m[1]*self.l[0]*self.l[1]*s1*dq[0]**2 + 0.5*self.m[1]*self.g*self.l[1]*s01
        b = np.zeros(T*self.ndof)
        b[::self.ndof] = b0
        b[1::self.ndof] = b1
        return b

    def b(self, w):
        return self.b_((self.Phi @ w).reshape(len(self.t_cost_array), self.ndof).T, (self.dPhi @ w).reshape(len(self.t_cost_array), self.ndof).T)

    def forward_dynamics(self, q, dq, tau):
        return np.linalg.pinv(self.M_(q)) @ (tau - self.b_(q, dq))

    def inverse_dynamics(self, w):
        return (self.M(w) @ self.ddPhi @ w + self.b(w)).reshape(len(self.t_cost_array), self.ndof)

    def plot_robot(self, ax, q, c):
        link_plot, = ax.plot([0.0, self.l[0]*np.sin(q[0]), self.l[0]*np.sin(q[0])+self.l[1]*np.sin(q[0]+q[1])], [0.0, -self.l[0]*np.cos(q[0]), -self.l[0]*np.cos(q[0])-self.l[1]*np.cos(q[0]+q[1])], lw=2, color=c)
        mass_plot = ax.scatter([0.5*self.l[0]*np.sin(q[0]), self.l[0]*np.sin(q[0])+0.5*self.l[1]*np.sin(q[0]+q[1])], [-0.5*self.l[0]*np.cos(q[0]), -self.l[0]*np.cos(q[0])-0.5*self.l[1]*np.cos(q[0]+q[1])], s=100, color=c)
        return link_plot, mass_plot

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
        q = (self.Phi @ w).reshape(len(self.t_cost_array), self.ndof)
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$q$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, q, lw=self.lw_)
        plt.plot(self.t_cost_array, np.pi*np.ones_like(self.t_cost_array), 'k--')
        plt.tight_layout()

    def plot_dq(self, w):
        dq = (self.dPhi @ w).reshape(len(self.t_cost_array), self.ndof)
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\dot{q}$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, dq, lw=self.lw_)
        plt.tight_layout()

    def plot_ddq(self, w):
        ddq = (self.ddPhi @ w).reshape(len(self.t_cost_array), self.ndof)
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\ddot{q}$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, ddq, lw=self.lw_)
        plt.tight_layout()

    def plot_tau(self, w):
        plt.figure(figsize=(16, self.fig_height), dpi=120, facecolor='w', edgecolor='k')
        plt.xlabel(r'$t$', fontsize=self.fontsize_)
        plt.ylabel(r'$\tau$', fontsize=self.fontsize_)
        plt.plot(self.t_cost_array, self.inverse_dynamics(w), lw=self.lw_)
        plt.plot([0,self.T], [self.tau_lim, self.tau_lim], 'k--')
        plt.plot([0,self.T], [-self.tau_lim, -self.tau_lim], 'k--')
        plt.tight_layout()

    def save_animation(self, q, filename="animation", fps=25):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120, facecolor='w', edgecolor='k')
        if len(q) != len(self.t_cost_array):
            q = (self.Phi @ q).reshape(len(self.t_cost_array), self.ndof)
        t_extra = 1.0
        T = self.T + t_extra
        q = np.concatenate((np.repeat([self.q_0], int(0.5*t_extra/self.dt_cost), axis=0), q, np.repeat([self.q_d], int(0.5*t_extra/self.dt_cost), axis=0)))

        K = len(q)
        dt_ani = 1./fps

        # forward kinematics
        Xs = np.zeros((K, self.ndof+1, 2))
        Xs_mass = np.zeros((K, self.ndof+1, 2))

        for k in range(K):
            q_ = q[k]
            Xs[k] = np.array([[0.0, 0.0], [self.l[0]*np.sin(q_[0]), -self.l[0]*np.cos(q_[0])], [self.l[0]*np.sin(q_[0])+self.l[1]*np.sin(q_[0]+q_[1]), -self.l[0]*np.cos(q_[0])-self.l[1]*np.cos(q_[0]+q_[1])]])
            Xs_mass[k] = np.array([[0.0, 0.0], [0.5*self.l[0]*np.sin(q_[0]), -0.5*self.l[0]*np.cos(q_[0])], [self.l[0]*np.sin(q_[0])+0.5*self.l[1]*np.sin(q_[0]+q_[1]), -self.l[0]*np.cos(q_[0])-0.5*self.l[1]*np.cos(q_[0]+q_[1])]])
        self.plot_robot(ax, self.q_d, 'k')
        link_plot, joint_plot = self.plot_robot(ax, q[0], 'g')

        axes = plt.gca()
        axes.set_xlim([-1.6,1.6])
        axes.set_ylim([-1.6,1.6])

        def animate(i_):
            i = int(i_*dt_ani/self.dt_cost)
            link_plot.set_xdata(Xs[i,:,0])
            link_plot.set_ydata(Xs[i,:,1])
            joint_plot.set_offsets(Xs_mass[i,1:,:])
            return link_plot, joint_plot

        ani = animation.FuncAnimation(
        fig, animate, interval=dt_ani*1e3, blit=True, frames=int(T*fps), repeat=False)

        gif_name = r"/home/julius/Documents/python/pac/traj_opt/media/" + filename + ".gif" 
        writergif = animation.PillowWriter(fps=30) 
        ani.save(gif_name, writer=writergif)