from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym-light',
      version='0.0.1',
      author='Benjamin Abernathy',
      author_email='ben.abernathy@gmail.com',
      description='A light seeking environment for Gym',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/benabernathy/gym_light',
      install_requires=['gym', 'numpy'])
