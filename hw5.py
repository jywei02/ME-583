# ME 574 Spring 2025: HW5 template
# See detailed descriptions of Problems 1 and 2 below. 

import taichi as ti
import numpy as np
import time
ti.init(arch=ti.cuda, kernel_profiler=True, offline_cache=False)

# fields for problem 1
n = 1000
f = ti.field(dtype=float, shape=(n, n)) # values of f-rep defining function
g = ti.field(dtype=float, shape=(n, n)) # values of integrand
dI = ti.field(dtype=float, shape=(n, n)) # grid point contributions to integral
pixels = ti.field(dtype=float, shape=(n, n))
perim = ti.field(dtype=float, shape=())

# fields and variables for problem 2
# import taichi as ti
# ti.init(arch=ti.cuda) # Alternatively, ti.init(arch=ti.cpu)

N = 128
cell_size = 1.0 / N
gravity = 0.5
stiffness = 1600
damping = 2
dt = 5e-4

ball_radius = 0.2
ball_center = ti.Vector.field(3, float, (2,))

x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)

####################
#### Problem 1 #####
####################
# The code below is a taichi implementation of 2D grid-based integration.
# Your mission is to modify the code appropriately so that the computation
# of point contributions and shading assignments are fully parallelized 
# (over all grid points/pixels) except for `nested_paint()` which remains serial for comparison.
# Note that the code includes kernel profiling.
# Modify the print strings at the end of `p1()` to record 
# the profiling results on your system.
###################

h = 1./(n-1)
eps = 1e-10
xc, yc = 0.5, 0.5
delta = 0.005
w = 0.25   
dw = 1.05

# The functions below define some shapes for testing.
@ti.func
def circ(i, j, x0, y0, r):
	x, y = i*h, j*h
	return ti.math.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)) - r

@ti.func
def block(i, j, x0, y0, r):
	x, y = i*h, j*h
	return ti.math.max(abs((x-x0)), abs((y-y0))) - 0.125

@ti.func
def cut(i, j, x0, y0, r):
	x, y = i*h, j*h
	return ti.math.max(circ(i,j,x0,y0,r), -block(i,j,x0,y0,r))

@ti.func
def one(i, j):
	return 1. # to compute perimeter

@ti.func
def chi(val):
	return 1. if val<0 else 0.

@ti.func
def point_contrib(i, j, f, g, h):
	val = 0.
	if 0<i<n-1 and 0<j<n-1:
		east, west = f[i+1,j], f[i-1,j]
		north, south = f[i, j+1], f[i,j-1]
		f_x, f_y = (east - west)/(2.*h), (north - south)/(2.*h)
		chi_x, chi_y = (chi(east) - chi(west))/(2.*h), (chi(north) - chi(south))/(2.*h)
		denom2 = f_x*f_x + f_y*f_y
		numer = g[i,j] * (f_x*chi_x + f_y*chi_y)
		if ti.abs(denom2)>eps:
			val = -numer/ti.math.sqrt(denom2) #left off h*h
	return val

@ti.kernel
def g_kernel(integrand:ti.template()):
	for i in range(n):
		for j in range(n):
			g[i,j] = integrand(i,j)

@ti.kernel
def f_kernel(frep:ti.template(), x0:float, y0:float, r:float):
	for i in range(n):
		for j in range(n):
			f[i,j] = frep(i,j, x0, y0, r)

@ti.kernel
def nested_paint(x: float, y: float, w: float):
	for i in range(1,n-1):
		for j in range(1,n-1):
			dI[i, j] = point_contrib(i, j, f, g, h)

@ti.kernel
def paint(x: float, y: float, w: float):
    for i, j in ti.ndrange(n, n):
        if i <= 0 or i>= n-1 or j<=0 or j>=n-1:
            dI[i, j] = point_contrib(i, j, f, g, h)
        else:
            dI[i, j] = 0.0

@ti.kernel
def reduce(f:ti.template()):
	perim[None]=0.
	for i in range(n):
		for j in range(n):
			perim[None] += f[i,j]

def p1():
	g_kernel(one)
	f_kernel(block,xc,yc,w)
	start_serial = time.time()
	nested_paint(xc,yc,w)
	ti.sync()
	serial_time = time.time() - start_serial
	start_parallel = time.time()
	paint(xc,yc,w)
	ti.sync()
	parallel_time = time.time() - start_parallel
	reduce(dI)
	
	print('Problem 1')
	print(f"perim = {h*h*perim[None]:.2f}\n")
	print(f'Original timing = {serial_time*1000:.2f} ms')
	print(f'Fully parallel timing = {parallel_time*1000:.2f} ms')
	print(f'Speedup factor = {serial_time/parallel_time:.2f}')

####################
#### Problem 2 #####
####################
# The code below (along with the field definitions above) is the 
# 'head first taichi` example simulating a cloth falling on a sphere.
# Your task is to modify the code in 3 ways:
# a) Have the cloth fall on 2 spheres with centers 
# at {0.25, -0.5, 2} and {0.75, -0.5, 2}.
# b) Modify the interaction so when a point on the cloth hits the sphere
# it can slide instead of sticking; i.e. zero out only the velocity
# component normal to the surface of the sphere for points moving downward into the sphere.
# c) Modify the scene so that the ball is not seen penetrating through the cloth.
# A simple approach (like a small reduction in the radius of the rendered sphere) is fine.
###################


def init_scene():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([i * cell_size ,
                             j * cell_size / ti.sqrt(2),
                             (N - j) * cell_size / ti.sqrt(2)])
    ball_center[0] = ti.Vector([0.25, -0.5, 0.0])
    ball_center[1] = ti.Vector([0.75, -0.5, 0.0])

@ti.kernel
def set_indices():
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j

links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = [ti.Vector(v) for v in links]

@ti.kernel
def step():
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
    for i in ti.grouped(x):
        force = ti.Vector([0.0,0.0,0.0])
        for d in ti.static(links):
            j = min(max(i + d, 0), [N-1,N-1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i-j).norm()
            if original_length != 0:
                force +=  stiffness * relative_pos.normalized() * (current_length - original_length) / original_length
        v[i] +=  force * dt
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)

        xi = x[i]
        vi = v[i]
        for k in range(ball_center.shape[0]):
            center = ball_center[k]
            diff = xi - center
            dist = diff.norm()
            if dist <= ball_radius:
                normal = diff / dist
                v_dot_n = vi.dot(normal)
                if v_dot_n < 0:
                    vi -= v_dot_n * normal
        
        x[i] += dt * v[i]
        v[i] = vi


				
@ti.kernel
def set_vertices():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]

def p2():
	init_scene()
	set_indices()
	window = ti.ui.Window("Cloth", (800, 800), vsync=True)
	canvas = window.get_canvas()
	scene = ti.ui.Scene()
	camera = ti.ui.make_camera()

	while window.running:
		for i in range(30):
			step()
		set_vertices()

		camera.position(0.5, -0.5, 2)
		camera.lookat(0.5, -0.5, 0)
		scene.set_camera(camera)

		scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
		scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5), two_sided = True)
		scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0, 0))
		canvas.scene(scene)
		window.show()

def main():
	p1()
	p2()

if __name__ == '__main__':
	main()
 

