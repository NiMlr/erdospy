from setuptools import setup

setup(name='erdospy',
      version='0.1',
      description='Sample the `G(n, m)`-model of Erdos–Renyi random graphs.',
      url='http://github.com/NiMlr/erdospy',
      author='Nils Müller',
      license='MIT',
      packages=['erdospy', 'erdospy.test'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      zip_safe=False)
