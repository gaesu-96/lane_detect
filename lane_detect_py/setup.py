from setuptools import find_packages, setup

package_name = 'lane_detect_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gyesu96',
    maintainer_email='gyesu96@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detect = lane_detect_py.lane_detect:main',
            'manipulate = lane_detect_py.manipulate:main',
            'lane_detect_dl = lane_detect_py.lane_detect_dl:main',
        ],
    },
)
