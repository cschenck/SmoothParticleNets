
"""
An example of a fluid simulation built using this library. The FluidSim class below implements
Position Based Fluids. The main function sets up and runs the simulation. Unfortunately this
library does not come with any visualization tools, so it only prints the average position of
the particles. However it should be easy to hook up to an outside viewer if so desired.

This file takes no command line arguments, so you can just run it as is.
"""

import numpy as np

import SmoothParticleNets as spn
import torch
from torch.autograd import Variable
import torch.nn as nn

import pdb

NSUBSTEPS = 1
DT = 1.0/60
STIFFNESS = 2.99e-11
DENSITY_REST = 17510.1
GRAVITY = [0, -9.8, 0]
MAX_VEL = 0.5*0.1*NSUBSTEPS/DT
COHESION = 0.1
VISCOSITY = 60.0
SURFACE_TENSION = 0.0
SURFACE_CONSTRAINT_SCALE = 168628.0
MAX_ACC = 0.833
NUM_ITERATIONS = 3
RELAXATION = 1.0
DAMP = 1.0
COLLISION_DISTANCE = 0.00125
NUM_STATIC_ITERATIONS = 1
NUM_FIX_ITERATIONS = NUM_STATIC_ITERATIONS
FLUID_REST_DISTANCE = 0.55

# A helper class to create a set of ConvSDF layers with different sized kernels for
# computing numerical gradients.


class GradConvSDF(nn.Module):
    def __init__(self, sdfs, sdf_sizes, ndim, max_distance):
        super(GradConvSDF, self).__init__()
        self.ndim = ndim
        self.convsdfgrad = []
        for d in range(ndim):
            ks = [1]*ndim
            ks[d] = 3
            kernel_size = ks
            dilation = 0.0005
            convsdf = spn.ConvSDF(sdfs, sdf_sizes, 1, ndim, kernel_size=ks, dilation=dilation,
                                  max_distance=max_distance, with_params=False, compute_pose_grads=True)
            convsdf.weight.data.fill_(0)
            convsdf.weight.data[0, 0] = -1/(2*dilation)
            convsdf.weight.data[0, 2] = 1/(2*dilation)
            convsdf.bias.data.fill_(0)
            self.convsdfgrad.append(convsdf)
            exec("self.convsdfgrad%d = convsdf" % d)

    def forward(self, locs, idxs, poses, scales):
        return torch.cat([self.convsdfgrad[d](locs, idxs, poses, scales)
                          for d in range(self.ndim)], 2)


class FluidSim(nn.Module):
    def __init__(self, sdfs, sdf_sizes, radius, ndim, with_params=[], init_params={}):
        """
        Initializes a fluid simulation model. 

        Arguments:
            sdfs: A list of SDFs for all objects in the simulation. This argument is passed directly as the sdfs
                  argument of ConvSDF. Refer to the documentation of that layer for details.
            sdf_sizes: A list of the sizes for the sdfs argument. This argument is also passed directy to ConvSDF.
                       Refer to the documentation for that layer for details.
            radius: The particle interaction radius. Particles that are further than this apart do not
                    interact. Larger values for this parameter slow the simulation significantly.
            ndim: The dimensionality of the coordinate space in the simulation. Only 2 or 3 is supported, and only
                  3 has been tested.
            with_params: (optional) List of parameter names. These will be the trainabale parameters in this 
                         module. See below for a list of parameters. Only parameters labelled as trainable may
                         appear in this list.
            init_params: A dictionary mapping parameter names to values to initialize them with. See below for a
                         list of parameters. If a parameter does not appear in this list, it is initialized with
                         its default value. Default values are listed at the top of the file.

        Parameters:
            nSubsteps: Each call to forward will divide the simulation time by this value and run it this many
                       times. Useful when it is desired to call the sim multiple times per timestep.
            dt: The amount of time that elapses during one forward call. This is used to scale the various
                simulation parameters.
            stiffness: A parameter controlling how the pressure offset is computed. It is not recommended to
                       change this from the default.
            maxSpeed: The maximum magnitude of the particle velocities. Higher velocities are clamped.
            cohesion[trainable]: The cohesion constant. 
            viscosity[trainable]: The viscosity constant.
            surfaceTension[trainable]: The surface tension constant.
            numIterations: The number of constraint solver iterations to do per simulation step.
            relaxationFactor: The value to divide the constraint solution deltas by to "relax" the solver.
            damp: The factor to dampen the constraint solution deltas by.
            collisionDistance: When a particle is closer than this to an object, it is considered colliding.
            numStaticIterations: When moving particles or objects, this is the number of substeps that 
                                 movement is split into to check for collisions. The higher this value, the
                                 less likely it is that particles will clip through objects but the slower
                                 the simulation is.
            fluidRestDistance: The distancce fluid particles should be from each other at rest, expressed
                               as a ratio of the radius.

        """
        super(FluidSim, self).__init__()

        self.all_params = {
            "nSubsteps": NSUBSTEPS,
            "dt": DT,
            "stiffness": STIFFNESS,
            "density_rest": DENSITY_REST,
            "maxSpeed": MAX_VEL,
            "cohesion": COHESION,
            "viscosity": VISCOSITY,
            "surfaceTension": SURFACE_TENSION,
            "surfaceConstraintScale": SURFACE_CONSTRAINT_SCALE,
            "maxAcceleration": MAX_ACC,
            "numIterations": NUM_ITERATIONS,
            "relaxationFactor": RELAXATION,
            "damp": DAMP,
            "collisionDistance": COLLISION_DISTANCE,
            "numStaticIterations": NUM_STATIC_ITERATIONS,
            "numFixIterations": NUM_FIX_ITERATIONS,
            "fluidRestDistance": FLUID_REST_DISTANCE*radius,
        }

        self.radius = radius
        self.ndim = ndim
        self._calculate_rest_density(init_params["fluidRestDistance"]
                                     if "fluidRestDistance" in init_params else self.all_params["fluidRestDistance"])

        max_distance = max([sdf.max().item()
                            for sdf in sdfs]) if len(sdfs) else 1e5
        if not len(sdfs):
            sdfs = [torch.from_numpy(
                np.zeros([1]*self.ndim, dtype=np.float32)), ]
            sdf_sizes = [1, ]

        self.coll = spn.ParticleCollision(
            ndim, radius, max_collisions=128, include_self=False)
        self.reorder_un2sort = spn.ReorderData(reverse=False)
        self.reorder_sort2un = spn.ReorderData(reverse=True)

        self.convsdfgrad = GradConvSDF(
            sdfs, sdf_sizes, ndim, max_distance=max_distance)
        self.convsdfcol = spn.ConvSDF(sdfs, sdf_sizes, 1, ndim, 1, 1,
                                      max_distance=max_distance, with_params=False, compute_pose_grads=True)
        self.convsdfcol.weight.data.fill_(-1)
        self.convsdfcol.bias.data.fill_(0)
        self.relu = nn.ReLU()

        layer_types = [
            ('dspiky', 1, True),
            ('dspiky', ndim, True),
            ('constant', 1, False),
            ('constant', ndim, False),
            ('spiky', 1, False),
            ('spiky', ndim, False),
            ('cohesion', 1, True),
            ('cohesion', ndim, True),
        ]

        for kernel, dim, normed in layer_types:
            conv = spn.ConvSP(dim, dim, ndim, kernel_size=1, dilation=1, radius=radius,
                              dis_norm=normed, with_params=False, kernel_fn=kernel)
            conv.bias.data.fill_(0)
            conv.weight.data.fill_(0)
            for i in range(dim):
                conv.weight.data[i, i, 0] = 1
            exec("self.%s%s%s = conv" % (kernel, "D" if dim == ndim else str(dim),
                                         "normd" if normed else ""))

        self.register_buffer("ones", Variable(torch.zeros(1)))

        self.register_buffer("gravity", Variable(torch.from_numpy(
            np.array(init_params["gravity"] if "gravity" in init_params else GRAVITY,
                     dtype=np.float32).reshape((1, 1, -1))), requires_grad=False))
        self.unroll = 0

        self.param_dict = {}
        for p, init_val in self.all_params.items():
            if init_params is not None and p in init_params:
                init_val = init_params[p]
            self.__setattr__("_"+p, init_val)
            if p in with_params:
                v = 1.0
                self.register_parameter(p, torch.nn.Parameter(torch.from_numpy(
                    np.array([v], dtype=np.float32))))

    def _calculate_rest_density(self, fluidRestDistance):
        points = np.array(self._tight_pack3D(
            self.radius, fluidRestDistance, 2048))
        d = np.sqrt(np.sum(points**2, axis=1))
        rho = 0
        rhoderiv = 0
        for dd in d:
            rho += spn.KERNEL_FN["spiky"](dd, self.radius)
            rhoderiv += spn.KERNEL_FN["dspiky"](dd, self.radius)**2
        self.all_params["density_rest"] = float(rho)
        self.all_params["stiffness"] = float(1.0/rhoderiv)

    # Generates an optimally dense sphere packing at the origin (implicit sphere at the origin)
    def _tight_pack3D(self, radius, separation, maxPoints):
        dim = int(np.ceil(1.0*radius/separation))

        points = []

        for z in range(-dim, dim+1):
            for y in range(-dim, dim+1):
                for x in range(-dim, dim+1):
                    xpos = x*separation + \
                        (separation*0.5 if ((y+z) & 1) else 0.0)
                    ypos = y*np.sqrt(0.75)*separation
                    zpos = z*np.sqrt(0.75)*separation

                    # skip center
                    if xpos**2 + ypos**2 + zpos**2 == 0.0:
                        continue

                    if len(points) < maxPoints and np.sqrt(xpos**2 + ypos**2 + zpos**2) <= radius:
                        points.append([xpos, ypos, zpos])
        return points

    def SetSDFs(self, sdfs, sdf_sizes):
        if not len(sdfs):
            sdfs = [torch.from_numpy(
                np.zeros([1]*self.ndim, dtype=np.float32)), ]
            sdf_sizes = [1, ]
        self.convsdfcol.SetSDFs(sdfs, sdf_sizes)
        for i in range(self.ndim):
            layer = getattr(self.convsdfgrad, "convsdfgrad%d" % i)
            layer.SetSDFs(sdfs, sdf_sizes)

    def _cap_magnitude(self, A, cap):
        d = len(A.size())
        vv = torch.norm(A, 2, d-1, keepdim=True)
        vv = cap/(vv + 0.0001)
        vv = -(self.relu(-vv + 1.0) - 1.0)
        return A*vv

    def _interp_poses(self, last_poses, poses, t):
        lq = last_poses[:, :, -4:]
        pq = poses[:, :, -4:]
        lt = last_poses[:, :, :-4]
        pt = poses[:, :, :-4]
        # Compute the quaternion dot product.
        dot = torch.sum(lq*pq, 2, keepdim=True).data
        dot[dot < 0] = -1
        dot[dot >= 0] = 1
        dot = Variable(dot, requires_grad=False)
        # Multiply negative dot product quaternions by -1.
        q = pq*dot
        # Linearly interpolate.
        rt = (pt - lt)*t + lt
        rq = (pq - lq)*t + lq
        q = rq/torch.sqrt(torch.sum(rq**2, 2, keepdim=True))
        ret = torch.cat((rt, rq), 2)

        return ret

    def _fix_static_collisions(self, locs, idxs, poses, scales, collisionDistance):
        ret = locs

        mtd = self.convsdfcol(ret, idxs, poses, scales) + collisionDistance
        intersect = self.relu(
            mtd) + self.relu(-self.relu(-(mtd - 0.5)) + 0.5)*0.0
        sdfgrad = self.convsdfgrad(ret, idxs, poses, scales)
        sdfgrad = torch.nn.functional.normalize(sdfgrad, dim=2, eps=1e-5)
        ret = ret + intersect*sdfgrad
        return ret

    def fixStaticCollisions(self, locs, new_locs, idxs, poses, scales):
        for p in self.all_params:
            if hasattr(self, p):
                scale = getattr(self, p)
            else:
                scale = 1
            #globals()[p] = getattr(self, "_"+p)*scale
            val = getattr(self, "_"+p)*scale
            exec("%s = val" % p)
        delta = (new_locs - locs)/numStaticIterations
        for _ in range(numStaticIterations):
            locs = locs + delta
            locs = self._fix_static_collisions(locs, idxs, poses,
                                               scales, collisionDistance)
        return locs
    
    def forward(self, locs, vel, idxs, poses, last_poses, scales, extra_constraints=None):
        """
        Compute one forward timestep of the fluid simulation. It takes as input the current
        fluid state (locs, vel) as well as the state of rigid objects in the scene. The return
        is the next fluid state. Note that the rigid objects are treated as static, i.e., the
        simulator doesn't move them.

        Inputs:
            locs: A BxNxD tensor where B is the batch size, N is the number of particles, and
                  D is the dimensionality of the coordinate space. The tensor contains the
                  locations of every particle.
            vel: A BxNxD tensor containing the velocities of every particle.
            idxs: A BxM tensor with the indices of every object in the scene. The indices are
                  in the sdfs list passed to the constructor. This is passed directly to
                  ConvSDF. Refer to that layer's documentation for more info.
            poses: A BxMxDD tensor with the pose of every object. This is passed directly to
                   ConvSDF. Refer there for more documentation.
            last_poses: A BxMxDD tensor containing the prior poses of the objects. This is
                        used to ensure that collisions between objects and particles are
                        resolved correctly.
            scales: A BxM tensor containing the scale of each object. This is passed directly
                    to ConvSDF. Refer there for more documentation.
            extra_constraints: (optional) A function that will be called last during the
                               constraint satisfaction loop. It takes as input a tensor of all
                               the particle locations and returns a tensor with a delta update
                               for every particle to better satisfy the constraint.

        Returns:
            locs: A BxNxD tensor with the updated positions of all the particles.
            vel: A BxNxD tensor with the updated velocities of all the particles.
        """

        for p in self.all_params:
            if hasattr(self, p):
                scale = getattr(self, p)
            else:
                scale = 1
            # globals()[p] = getattr(self, "_"+p)*scale
            val = getattr(self, "_"+p)*scale
            # exec("%s = val" % p)
            globals()[p] = val
        dt = self._dt
        dt /= nSubsteps
        
        if self.ones is None or self.ones.size()[:-1] != locs.size()[:-1]:
            with torch.no_grad():
                self.ones.resize_(locs.size()[:-1] + (1,)).fill_(1)

        
        _poses = last_poses
        for substep in range(nSubsteps):
            # pdb.set_trace()
            _last_poses = _poses
            _poses = self._interp_poses(
                last_poses, poses, 1.0*(1 + substep)/nSubsteps)
            # First move the objects and fix static collisions.
            new_locs = locs
            for static_iteration in range(numStaticIterations):
                t = 1.0*(1 + static_iteration)/numStaticIterations
                pp = self._interp_poses(_last_poses, _poses, t)
                new_locs = self._fix_static_collisions(new_locs, idxs, pp,
                                                       scales, collisionDistance)

            oldvel = vel
            # GRAVITY
            vel = vel + self.gravity*dt
            # Cap velocity.
            vel = self._cap_magnitude(vel, maxSpeed)
            #new_locs = new_locs + vel*dt
            for static_iteration in range(numStaticIterations):
                new_locs = new_locs + vel*dt/numStaticIterations
                new_locs = self._fix_static_collisions(new_locs, idxs, _poses,
                                                       scales, collisionDistance)

            new_locs, vel, pidxs, neighbors = self.coll(new_locs, vel)

            for iteration in range(numIterations):
                density = self.spiky1(new_locs, self.ones, neighbors)
                nj = self.dspikyDnormd(new_locs, new_locs, neighbors)
                ni = new_locs*self.dspiky1normd(new_locs, self.ones, neighbors)
                nij = ni - nj

                # PRESSURE
                # p_i = STIFFNESS*max(\rho_i - DEN_REST, 0)
                # \sum_j n*(p_i + p_j)*W(d)
                # \sum_j (x_j - x_i)/d*(p_i + p_j)*W(d)
                # \sum_j(x_j/d*p_i*W(d)) + \sum_j(x_j/d*p_j*W(d)) -
                #      \sum_j(x_i/d*p_i*W(d)) - \sum_j(x_i/d*p_j*W(d))
                # p_i*\sum_j(x_j/d*W(d)) + \sum_j(x_j/d*p_j*W(d)) -
                #      x_i*p_i*\sum_j(1/d*W(d)) - x_i*\sum_j(1/d*p_j*W(d))
                pressure = stiffness*self.relu(density - density_rest)
                njp = self.dspikyDnormd(new_locs, new_locs*pressure, neighbors)
                nip = new_locs*self.dspiky1normd(new_locs, pressure, neighbors)
                nijp = nip - njp
                delta = -(pressure*nij + nijp)

                # COHESION
                nj = self.cohesionDnormd(new_locs, new_locs, neighbors)
                ni = new_locs * \
                    self.cohesion1normd(new_locs, self.ones, neighbors)
                nij = ni - nj
                delta = delta + -cohesion*nij*self.radius

                # SURFACE TENSION
                normals = nij*surfaceTension/density_rest/surfaceConstraintScale
                ncount = self.constant1(new_locs, self.ones, neighbors)
                delta = delta + \
                    (self.constantD(new_locs, normals, neighbors) - normals*ncount)

                # Extra constraints if there are any.
                if extra_constraints is not None:
                    delta += extra_constraints(new_locs)

                # Apply relaxation to the delta.
                scale = ncount/(1.0 + relaxationFactor)
                scale = self.relu(scale - damp) + damp
                delta = delta/scale
                #new_locs = new_locs + delta

                # STATIC COLLISIONS
                for static_iteration in range(numStaticIterations):
                    new_locs = new_locs + delta/numStaticIterations
                    new_locs = self._fix_static_collisions(new_locs, idxs, _poses,
                                                           scales, collisionDistance)

            # Correct velocity based on actualy movement.
            vel = (new_locs - self.reorder_un2sort(pidxs, locs))/dt

            # VISCOSITY
            vj = self.spikyD(new_locs, vel, neighbors)
            vi = vel*self.spiky1(new_locs, self.ones, neighbors)
            vji = vj - vi
            vel = vel + dt*viscosity/density_rest*vji

            new_locs, vel = self.reorder_sort2un(pidxs, new_locs, vel)

            # Cap acceleration.
            # This causes unstable gradients, leave commented out.
            # vdelta = vel - oldvel
            # vdelta = self._cap_magnitude(vdelta, MAX_ACC)
            # vel = oldvel + vdelta
            # new_locs = self._check_for_clipping(locs, new_locs, idxs, poses, scales,
            #     numFixIterations)
            locs = new_locs

        return locs, vel

    def load(self, state_dict):
        self.load_state_dict(state_dict)


# Generate an inside-out box to act as a boundry for the particles.
def GenerateBoundsSDF(bounds):
    dx = bounds[:, 1] - bounds[:, 0]
    # Add a margin on each side.
    MARGIN = 0.5
    DIM = 300
    cellsize = ((1.0 + 2*MARGIN)*dx).min()/DIM
    shape = np.ceil((1.0 + 2*MARGIN)*dx/cellsize).astype(np.int)
    sdf = np.ones(shape, dtype=np.float32)*dx.max()
    position = bounds[:, 0] - MARGIN*dx
    for d in range(bounds.shape[0]):
        xx = np.arange(position[d] + cellsize/2,
                       position[d] + shape[d]*cellsize, cellsize)
        minx = xx - bounds[d, 0]
        maxx = bounds[d, 1] - xx
        xx = np.minimum(minx, maxx)
        s = [1]*bounds.shape[0]
        s[d] = xx.shape[0]
        xx = xx.reshape(s)
        sdf = np.minimum(sdf, xx)
    return {
        'sdf': sdf.astype(np.float32),
        'sdf_size': cellsize,
        'position': position,
    }


def main():
    bounds = np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 1.0]])
    bounds_sdf = GenerateBoundsSDF(bounds)
    spnet = FluidSim([torch.from_numpy(bounds_sdf["sdf"])], [
                     bounds_sdf["sdf_size"]], radius=0.1, ndim=3)
    spnet = spnet.cuda()
    
    locs = torch.rand(1, 1000, 3).cuda()
    locs[:, :, 1] += 4.0
    vel = torch.zeros(1, 1000, 3).cuda()
    idxs = torch.zeros(1, 1).cuda()
    scales = torch.ones(1, 1).cuda()
    obj_poses = torch.zeros(1, 1, 7).cuda()
    obj_poses[:, :, -1] = 1.0
    while True:
        # pdb.set_trace()
        locs, vel = spnet(locs, vel, idxs, obj_poses, obj_poses, scales)
        print(locs[0, ...].mean(0))


if __name__ == '__main__':
    main()
