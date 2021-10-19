# PINN_Implicit_SDF

Generate 3D meshes based on SDFs (signed distance functions) with a
dirt simple Python API.

Special thanks to [Inigo Quilez](https://iquilezles.org/) for his excellent documentation on signed distance functions:

- [3D Signed Distance Functions](https://iquilezles.org/www/articles/distfunctions/distfunctions.htm)
- [2D Signed Distance Functions](https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm)

## Example

Here is a complete example that generates the model shown. This is the
canonical [Constructive Solid Geometry](https://en.wikipedia.org/wiki/Constructive_solid_geometry)
example. Note the use of operators for union, intersection, and difference.

```python
from sdf import *

f = sphere(1) & box(1.5)

c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)

f.save('out.stl')
```

Yes, that's really the entire code! You can 3D print that model or use it
in a 3D application.

## More Examples

Have a cool example? Submit a PR!

| [gearlike.py](examples/gearlike.py) | [knurling.py](examples/knurling.py) | [blobby.py](examples/blobby.py) | [weave.py](examples/weave.py) |
| --- | --- | --- | --- |
| ![gearlike](docs/images/gearlike.png) | ![knurling](docs/images/knurling.png) | ![blobby](docs/images/blobby.png) | ![weave](docs/images/weave.png) |
| ![gearlike](docs/images/gearlike.jpg) | ![knurling](docs/images/knurling.jpg) | ![blobby](docs/images/blobby.jpg) | ![weave](docs/images/weave.jpg) |

## Requirements

Note that the dependencies will be automatically installed by setup.py when
following the directions below.

- Python 3
- matplotlib
- meshio
- numpy
- Pillow
- scikit-image
- scipy

## Installation

Use the commands below to clone the repository and install the `sdf` library
in a Python virtualenv.

```bash
git clone https://github.com/fogleman/sdf.git
cd sdf
virtualenv env
. env/bin/activate
pip install -e .
```

Confirm that it works:

```bash
python examples/example.py # should generate a file named out.stl
```

You can skip the installation if you always run scripts that import `sdf`
from the root folder.

## File Formats

`sdf` natively writes binary STL files. For other formats, [meshio](https://github.com/nschloe/meshio)
is used (based on your output file extension). This adds support for over 20 different 3D file formats,
including OBJ, PLY, VTK, and many more.

## Viewing the Mesh

Find and install a 3D mesh viewer for your platform, such as [MeshLab](https://www.meshlab.net/).

I have developed and use my own cross-platform mesh viewer called [meshview](https://github.com/fogleman/meshview) (see screenshot).
Installation is easy if you have [Go](https://golang.org/) and [glfw](https://www.glfw.org/) installed:

```bash
$ brew install go glfw # on macOS with homebrew
$ go get -u github.com/fogleman/meshview/cmd/meshview
```

Then you can view any mesh from the command line with:

```bash
$ meshview your-mesh.stl
```

See the meshview [README](https://github.com/fogleman/meshview) for more complete installation instructions.

On macOS you can just use the built-in Quick Look (press spacebar after selecting the STL file in Finder) in a pinch.

# API

In all of the below examples, `f` is any 3D SDF, such as:

```python
f = sphere()
```

## Bounds

The bounding box of the SDF is automatically estimated. Inexact SDFs such as
non-uniform scaling may cause issues with this process. In that case you can
specify the bounds to sample manually:

```python
f.save('out.stl', bounds=((-1, -1, -1), (1, 1, 1)))
```

## Resolution

The resolution of the mesh is also computed automatically. There are two ways
to specify the resolution. You can set the resolution directly with `step`:

```python
f.save('out.stl', step=0.01)
f.save('out.stl', step=(0.01, 0.02, 0.03)) # non-uniform resolution
```

Or you can specify approximately how many points to sample:

```python
f.save('out.stl', samples=2**24) # sample about 16M points
```

By default, `samples=2**22` is used.

*Tip*: Use the default resolution while developing your SDF. Then when you're done,
crank up the resolution for your final output.

## Batches

The SDF is sampled in batches. By default the batches have `32**3 = 32768`
points each. This batch size can be overridden:

```python
f.save('out.stl', batch_size=64) # instead of 32
```

The code attempts to skip any batches that are far away from the surface of
the mesh. Inexact SDFs such as non-uniform scaling may cause issues with this
process, resulting in holes in the output mesh (where batches were skipped when
they shouldn't have been). To avoid this, you can disable sparse sampling:

```python
f.save('out.stl', sparse=False) # force all batches to be completely sampled
```

## Worker Threads

The SDF is sampled in batches using worker threads. By default,
`multiprocessing.cpu_count()` worker threads are used. This can be overridden:

```python
f.save('out.stl', workers=1) # only use one worker thread
```

## Without Saving

You can of course generate a mesh without writing it to an STL file:

```python
points = f.generate() # takes the same optional arguments as `save`
print(len(points)) # print number of points (3x the number of triangles)
print(points[:3]) # print the vertices of the first triangle
```

If you want to save an STL after `generate`, just use:

```python
write_binary_stl(path, points)
```

## Visualizing the SDF

You can plot a visualization of a 2D slice of the SDF using matplotlib.
This can be useful for debugging purposes.

```python
f.show_slice(z=0)
f.show_slice(z=0, abs=True) # show abs(f)
```

You can specify a slice plane at any X, Y, or Z coordinate. You can
also specify the bounds to plot.

Note that `matplotlib` is only imported if this function is called, so it
isn't strictly required as a dependency.

<br clear="right">

## How it Works

The code simply uses the [Marching Cubes](https://en.wikipedia.org/wiki/Marching_cubes)
algorithm to generate a mesh from the [Signed Distance Function](https://en.wikipedia.org/wiki/Signed_distance_function).

This would normally be abysmally slow in Python. However, numpy is used to
evaluate the SDF on entire batches of points simultaneously. Furthermore,
multiple threads are used to process batches in parallel. The result is
surprisingly fast (for marching cubes). Meshes of adequate detail can
still be quite large in terms of number of triangles.

The core "engine" of the `sdf` library is very small and can be found in
[mesh.py](https://github.com/fogleman/sdf/blob/main/sdf/mesh.py).

In short, there is nothing algorithmically revolutionary here. The goal is
to provide a simple, fun, and easy-to-use API for generating 3D models in our
favorite language Python.

## Files

- [sdf/d2.py](https://github.com/fogleman/sdf/blob/main/sdf/d2.py): 2D signed distance functions
- [sdf/d3.py](https://github.com/fogleman/sdf/blob/main/sdf/d3.py): 3D signed distance functions
- [sdf/dn.py](https://github.com/fogleman/sdf/blob/main/sdf/dn.py): Dimension-agnostic signed distance functions
- [sdf/ease.py](https://github.com/fogleman/sdf/blob/main/sdf/ease.py): [Easing functions](https://easings.net/) that operate on numpy arrays. Some SDFs take an easing function as a parameter.
- [sdf/mesh.py](https://github.com/fogleman/sdf/blob/main/sdf/mesh.py): The core mesh-generation engine. Also includes code for estimating the bounding box of an SDF and for plotting a 2D slice of an SDF with matplotlib.
- [sdf/progress.py](https://github.com/fogleman/sdf/blob/main/sdf/progress.py): A console progress bar.
- [sdf/stl.py](https://github.com/fogleman/sdf/blob/main/sdf/stl.py): Code for writing a binary [STL file](https://en.wikipedia.org/wiki/STL_(file_format)).
- [sdf/text.py](https://github.com/fogleman/sdf/blob/main/sdf/text.py): Generate 2D SDFs for text (which can then be extruded)
- [sdf/util.py](https://github.com/fogleman/sdf/blob/main/sdf/util.py): Utility constants and functions.

## SDF Implementation

It is reasonable to write your own SDFs beyond those provided by the
built-in library. Browse the SDF implementations to understand how they are
implemented. Here are some simple examples:

```python
@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return np.linalg.norm(p - center, axis=1) - radius
    return f
```

An SDF is simply a function that takes a numpy array of points with shape `(N, 3)`
for 3D SDFs or shape `(N, 2)` for 2D SDFs and returns the signed distance for each
of those points as an array of shape `(N, 1)`. They are wrapped with the
`@sdf3` decorator (or `@sdf2` for 2D SDFs) which make boolean operators work,
add the `save` method, add the operators like `translate`, etc.

```python
@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)
    return f
```

An SDF that operates on another SDF (like the above `translate`) should use
the `@op3` decorator instead. This will register the function such that SDFs
can be chained together like:

```python
f = sphere(1).translate((1, 2, 3))
```

Instead of what would otherwise be required:

```python
f = translate(sphere(1), (1, 2, 3))
```