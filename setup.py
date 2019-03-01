import os
import pathlib
from os.path import join as pjoin
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil

def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile
    
class CMakeExtension(Extension):

    def __init__(self, name, sources):
        # don't invoke the original build_ext for this special extension
        Extension.__init__(self, name, sources=sources)

class my_build_ext(build_ext):
    # This class allows C extension building to fail.
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

    def run(self):
        build_ext.run(self)

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        if os.path.isdir(str(build_temp.absolute())):
            shutil.rmtree(str(build_temp.absolute()))
        build_temp.mkdir(parents=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        # extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            # '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            # '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))        
        self.spawn(['cmake', os.path.join(str(cwd), 'PyCudaSampling')] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
            
        os.chdir(str(cwd))
        
    def build_python(self, ext):
        # extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        build_temp = pathlib.Path(self.build_temp)
        # ext.library_dirs.append(str(extdir.parent.absolute()))
        ext.library_dirs.append(str(build_temp.absolute()))
        build_ext.build_extension(self, ext)
        
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake(ext)
        else:
            self.build_python(ext)
      
CUDA = locate_cuda()

module1 = CMakeExtension('cudaSampling',
    sources=[
        os.path.join('PyCudaSampling', 'CMakeLists.txt'),
        os.path.join('PyCudaSampling', 'cudaCommon.h'),
        os.path.join('PyCudaSampling', 'cudaSampling_kernel.cu'),
        os.path.join('PyCudaSampling', 'cudaSampling.cu'),
        os.path.join('PyCudaSampling', 'cudaSampling.h'),
        ],
    )
  
module2 = Extension('PyCudaSampling',
    sources=[
        os.path.join('PyCudaSampling', 'PyCudaSampling.cpp'),
    ],
    extra_compile_args=['-fPIC'],
    include_dirs = ['./PyCudaSampling', CUDA['include']],
    library_dirs = [CUDA['lib64']],
    libraries=['cudaSampling', 'cudart'],
    )      
      
setup(name = 'PyCudaSampling',
    author = 'huangch', 
    author_email='huangch.tw@gmail.com',
    version = '0.0.2.dev201902141822',
    description="CUDA-based Image Sampling Tool.",
    ext_modules=[module1, module2],
    cmdclass = {'build_ext': my_build_ext},
    setup_requires=['numpy'],
    packages=['PyCudaSampling'],
    # package_data={'PyCudaSampling': ['libcudaSampling.so']},
    )
