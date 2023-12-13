import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def normalize(x):
    '''
    Vector normalization.

    **Parameters**

        x: * array *
            vector to be normalized

    **Returns**

        normalized x
    '''
    return x / np.linalg.norm(x)   # normalization


def generate_points_on_circle(radius, y_set, num_points):
    '''
    generate points on a circle centered (x=0, z=0).

    **Parameters**

        radius: * float *
            radius of the circle
        y_set: * float *
            given y value of points
        num_points: * int *
            number of points generated

    **Returns**

        coordinates: * list *
            a list of points generated
    '''
    # Generate equally spaced angles
    theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Calculate coordinates using the parameterized equations
    x_values = radius * np.cos(theta_values)
    y_values = y_set * np.ones_like(theta_values)  # y = 0.35
    z_values = radius * np.sin(theta_values)

    # Combine x, y, z coordinates into a single array
    coordinates = np.column_stack((x_values, y_values, z_values))

    return coordinates


def generate_pic(O, Q, k):
    '''
    generate picture given camera position and direction.

    **Parameters**

        O: * array *
            camera position
        Q: * array *
            camera direction
        k: * int *
            index of picture

    **Returns**

        None
    '''
    dir = np.copy(Q)
    img = np.zeros((h, w, 3))
    r = float(w) / h
    S = (-1. + dir[0], -1. / r + .25 + dir[1], 1. + dir[0], 1. / r + .25 + dir[1])

    print(O, dir, S)
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        # print("%.2f" % (i / float(w) * 100), "%")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            dir[:2] = (x, y)
            ray = Ray(O, normalize(dir - O))
            img[h - j - 1, i, :] = ray_scene.intersect_color(ray, 1)

    plt.imsave(f"sequence/pic{k}.png", img)


class Ray:
    '''
    This class handles the position and direction of the ray we trace

    **Parameters**
    origin: * array *
        ray startpoint (camera position)
    direction: * array *
        endpoint of the current ray
    '''
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)


class Object:
    '''
    This class handles the objects construct the virtual world

    **Parameters**
    position: * array *
        object position (center of sphere or a point on the plane)
    color: * array *
        RGB color of the object
    reflection: * float *
        Represents the amount of light that is reflected specularly (mirror-like reflection) from the surface.
    diffuse: * float *
        Represents the amount of light that is diffusely reflected from the surface. Diffuse reflection occurs
        when light is scattered in various directions.
    specular_c: * float *
        Represents the intensity of the specular reflection. It controls how "shiny" or "specular" the material
        appears. Lower values result in a broader and less intense specular highlight.
    specular_k: * float *
        Represents the shininess or smoothness of the material. Higher values result in a smaller and more concentrated
        specular highlight. This value is often referred to as the specular exponent or shininess coefficient.
    '''
    def __init__(self, position, color, reflection=0.85, diffuse=1.0, specular_c=0.6, specular_k=50):
        self.position = np.array(position)
        self.color = np.array(color)
        self.reflection = reflection
        self.diffuse = diffuse
        self.specular_c = specular_c
        self.specular_k = specular_k

    def intersect(self, ray):
        pass

    def get_normal(self, point):
        pass

    def get_color(self, point):  # return color of the object
        return self.color


class Sphere(Object):
    '''
    This class handles further features of sphere on the basis of Object

    **Parameters**
    radius: * float *
        radius of the sphere
    others: same as "Object" class
    '''
    def __init__(self, position, radius, color, **kwargs):
        super().__init__(position, color, **kwargs)
        self.radius = np.array(radius)

    def intersect(self, ray):
        '''
        Intersection test

        **Parameters**

            ray: * object *
                current ray to be traced

        **Returns**

            distance between intersection point and ray startpoint
        '''
        OC = self.position - ray.origin
        if np.linalg.norm(OC) < self.radius or np.dot(OC, ray.direction) < 0:
            return np.inf
        l = np.linalg.norm(np.dot(OC, ray.direction))
        m_square = np.linalg.norm(OC) ** 2 - l ** 2
        q_square = self.radius ** 2 - m_square
        return (l - np.sqrt(q_square)) if q_square >= 0 else np.inf

    def get_normal(self, point):
        return normalize(point - self.position)


class Plane(Object):
    '''
    This class handles further features of plane on the basis of Object

    **Parameters**
    normal: * array *
        normal vector of the plane
    others: same as "Object" class
    '''
    def __init__(self, position, normal, width=1.0, height=1.0, texture_file=None, reflection=0.15, diffuse=0.75,
                 specular_c=0.3, specular_k=50, **kwargs):
        super().__init__(position, None, reflection=reflection, diffuse=diffuse, specular_c=specular_c,
                         specular_k=specular_k, **kwargs)
        self.normal = np.array(normal)
        self.width = width
        self.height = height
        if texture_file:
            self.texture = self.load_texture(texture_file)
        else:
            self.texture = None

    def load_texture(self, texture_file):
        texture_image = Image.open(texture_file)
        return np.array(texture_image)

    def intersect(self, ray):
        dn = np.dot(ray.direction, self.normal)
        if np.abs(dn) < 1e-6:
            return np.inf
        d = np.dot(self.position - ray.origin, self.normal) / dn
        return d if d > 0 else np.inf

    def get_normal(self, point):
        return self.normal

    def get_color(self, point):
        if self.texture is not None:
            tex_coords = np.array([(point[0] + 0.5) / self.width, (point[2] + 0.5) / self.height])
            tex_coords *= np.array([self.texture.shape[1], self.texture.shape[0]])
            sampled_color = self.texture[int(tex_coords[1]) % self.texture.shape[0], int(tex_coords[0]) %
                                                                                     self.texture.shape[1]]
            return sampled_color / 255.0
        else:
            return self.color


class Scene:
    '''
    This class handles the scene (virtual world)

    **Parameters**
    objects: * list *
        a list of objects created
    light_point: * array *
        position of point light source
    light_color: * array *
        RGB color of point light source
    ambient: * float *
        intensity of ambient light
    '''
    def __init__(self, objects, light_point, light_color, ambient):
        self.objects = objects
        self.light_point = np.array(light_point)
        self.light_color = np.array(light_color)
        self.ambient = ambient

    def intersect_color(self, ray, intensity):
        min_distance = np.inf
        for obj in self.objects:
            current_distance = obj.intersect(ray)
            if current_distance < min_distance:
                min_distance, hit_object = current_distance, obj

        if min_distance == np.inf or intensity < 0.01:
            return np.array([0., 0., 0.])

        P = ray.origin + ray.direction * min_distance
        color = hit_object.get_color(P)
        N = hit_object.get_normal(P)
        PL = normalize(self.light_point - P)
        PO = normalize(ray.origin - P)

        c = self.ambient * color

        l = [obj.intersect(Ray(P + N * 0.0001, PL)) for obj in self.objects if obj != hit_object]
        if not (l and min(l) < np.linalg.norm(self.light_point - P)):
            c += hit_object.diffuse * max(np.dot(N, PL), 0) * color * self.light_color
            c += hit_object.specular_c * max(np.dot(N, normalize(PL + PO)), 0) ** hit_object.specular_k * self.light_color

        reflect_ray = ray.direction - 2 * np.dot(ray.direction, N) * N
        c += hit_object.reflection * self.intersect_color(Ray(P + N * 0.0001, reflect_ray), hit_object.reflection * intensity)

        return np.clip(c, 0, 1)


if __name__ == "__main__":
    # World construction #
    sphere1 = Sphere([0.75, 0.1, 1], 0.6, [1., 1., 1.])
    sphere2 = Sphere([-0.3, -0.2, 0.2], 0.3, [1., 0., 0.])
    sphere3 = Sphere([-2.75, 0., 3.5], 0.5, [0., 0., 1.])
    # sphere = Sphere([0., -0.2, 0.], 0.3, [1., 0., 0.])

    plane = Plane([0., -0.5, 0.], [0., 1., 0.], width=2.0, height=2.0, texture_file='jhu.jpeg')

    light_point = [5., 5., -10.]
    light_color = [1., 1., 1.]
    ambient = 0.05

    scene_objects = [sphere1, sphere2, sphere3, plane]
    # scene_objects = [sphere, plane]


    ##### 1 light rotation #####
    # light = generate_points_on_circle(radius=10, y_set=5, num_points=500)
    # O = np.array([0., 0.35, -1.])  # camera position
    # Q = np.array([0., 0., 0.])  # camera direction
    # w, h = 800, 600
    # k = 0
    # for light_point in light:
    #     k += 1
    #     ray_scene = Scene(scene_objects, light_point, light_color, ambient)
    #     generate_pic(O, Q, k)

    ##### 2 camera movement #####
    # ray_scene = Scene(scene_objects, light_point, light_color, ambient)
    # w, h = 800, 600
    #
    # O = np.array([-3., 0.35, -1.])  # camera position
    # Q = np.array([-3., 0., 0.])  # camera direction
    # for k in range(20):
    #     generate_pic(O, Q, k)
    #     O[0] += 0.2
    #     Q[0] += 0.2

    ##### 3 camera rotation #####
    ray_scene = Scene(scene_objects, light_point, light_color, ambient)
    w, h = 800, 600

    r1 = 7
    r2 = 4
    center = np.array([-1., 0., 1.75])  # camera center
    O = np.array([-1, 0.35, center[2] + r1])  # camera position
    Q = np.array([-1, 0.35, center[2] + r2])  # camera direction
    num = 1000  # number of picture to be rendered
    for k in range(num):
        generate_pic(O, Q, k)
        O[0] = center[0] + np.sin((k + 1) * 2 * np.pi / num) * r1
        O[2] = center[2] + np.cos((k + 1) * 2 * np.pi / num) * r1
        Q[0] = center[0] + np.sin((k + 1) * 2 * np.pi / num) * r2
        Q[2] = center[2] + np.cos((k + 1) * 2 * np.pi / num) * r2
