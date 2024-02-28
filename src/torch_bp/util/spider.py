import numpy as np


def make_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def make_tf_matrix(x, y, theta=0):
    tf = np.eye(3)
    tf[0:2, 0:2] = make_rot_matrix(theta)
    tf[0, 2] = x
    tf[1, 2] = y

    return tf


def normalize_angle(angle):
    """Normalize angle to stay between -PI and PI"""
    result = np.fmod(angle + np.pi, 2.0 * np.pi)
    if result <= 0:
        return result + np.pi
    return result - np.pi


class Spider(object):
    def __init__(self, x=0, y=0, qs=[],
                 n_arms=3, n_links_per_arm=2, arm_length=1,
                 root=None, links=None):
        self.n_arms = n_arms
        self.n_links_per_arm = n_links_per_arm
        self.n_joints = n_arms * n_links_per_arm

        self.arm_length = arm_length

        self.x = x
        self.y = y
        self.qs = qs

        self.links = [(self.x, self.y, 0)] + [None for _ in range(self.n_joints)]

        if len(qs) != self.n_joints:
            self.qs = self.random_init()
            self.update_links()

    def random_init(self, std=np.pi / 8):
        rot = np.random.uniform(0, 2 * np.pi / self.n_arms)  # Random rotation of the full body.
        arm_root_links = [i * (2 * np.pi / self.n_arms) + rot for i in range(self.n_arms)]
        init = arm_root_links + [0] * self.n_arms * (self.n_links_per_arm - 1)  # For each of the joints in the arm.
        init = [np.random.normal(q, std) for q in init]
        return init

    def set_state(self, x, y, qs=[]):
        self.x = x
        self.y = y

        if len(qs) <= len(self.links):
            for i in range(len(qs)):
                self.qs[i] = qs[i]

        self.update_links()

    def update_links(self):
        tw = make_tf_matrix(self.x, self.y)

        for i in range(self.n_joints):
            theta = self.qs[i]
            rect_tf = None
            if i < self.n_arms:
                # This is the first layer of joints, connected to the root.
                t1 = make_tf_matrix(self.arm_length, 0)
                rot1 = make_tf_matrix(0, 0, theta)
                rect_tf = rot1.dot(t1)
            else:
                # This is the second layer of joints, connected to the first layer.
                parent_joint = self.qs[i % self.n_arms]
                theta = normalize_angle(theta + parent_joint)

                t2 = make_tf_matrix(self.arm_length, 0)
                rot2 = make_tf_matrix(0, 0, self.qs[i])
                t1 = make_tf_matrix(self.arm_length, 0)
                rot1 = make_tf_matrix(0, 0, parent_joint)

                rect_tf = rot1.dot(t1).dot(rot2).dot(t2)

            pt = np.array([0, 0, 1]).reshape((3, 1))
            new_pt = tw.dot(rect_tf).dot(pt)

            self.links[i + 1] = (new_pt[0, 0], new_pt[1, 0], theta)

    def states(self):
        return np.array([link[:2] for link in self.links])
