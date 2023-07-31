from setuptools import setup, find_packages
import lbscratch

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                "finufft",
                'pfb-clean',

                "smoove"
                "@git+https://github.com/landmanbester/smoove.git"
                "@test_ci"
            ]


setup(
     name='lbscratch',
     version=lbscratch.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Misc utils",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/landmanbester/lbscratch",
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False,
     python_requires='>=3.8',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: POSIX :: Linux",
         "Topic :: Scientific/Engineering :: Astronomy",
     ],
     entry_points={'console_scripts':[
        'lbs = lbscratch.workers.main:cli'
        ]
     }


     ,
 )
