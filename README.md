# VIXM Algorithmic Trading Strategy
An algorithmic trading strategy incursion to create the first volatility security suitable for long term investors.

<div id="top"></div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a></li>
    <li><a href="#Model Specifications">Model Specifications</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
In crushes, when all assets go way down, the VIX goes up. The investment on the VIX would be ideal to diversify risk on those scary moments. However, nowadays all instruments available to invest in volatility are done through VIX futures. These instrument are called ETN (ie. VXX and VIXM), and are only suitable for high risk proffesional investors only, because of the price decay due to the rollover effect of the futures.

This project select one of these volatility vehicles and generate an strategy of daily trades. Using an adaboost classification model, the algorith classify the next day as a "positive return day" (signal 1) or a negative return day (signal 0) these way avoiding to invest on days expected to have a negative returns.

This project aims to generate a model with enough accuracy, so to enable an investment vehicle without the price decay, and so enable regular long term investors to invest in volatility.

This model that I present here is the machine learning algorith part of a larger project called ["VIXCOIN"](https://github.com/Fintech-Collaboration/vixx-token-dapp), which sell a token in the blockchain which return is connected to the return of the strategy. 

The first version of this project was to [predict the VIX Index](https://github.com/paocarvajal1912/vix_predictor). There we included a larger variaty of variables, including economical and google trends data. 

DISCLAIMER: Investment in our product is not an option yet. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
The adaboost machine learning algorith was built utilizing at least python 3.7 for the back end data analysis and machine learning 

Python Packages
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Datetime](https://pypi.org/project/DateTime/)
* [yfinance](https://pypi.org/project/yfinance/)
* [Sci-kit learn](https://scikit-learn.org/stable/)
* [Hvplot](https://hvplot.holoviz.org/)
* [Matplotlib](https://matplotlib.org/)
* [sys](https://docs.python.org/3/library/sys.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these steps below. 
Alternatively, you can jump straight to the demo section and test out the wesbite portion. 

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/Fintech-Collaboration/vixx-token-dapp.git
   ```
2. Install python packages listed in the Built With section. 
3. To test out the algorithmic trading strategy file, first go to your terminal. 
4. In your terminal, navigate to the location where the cloned repo resides.
5. Locate vixm_adaboost_model.ipynb and launch the file in jupyter notebook for data visualizations. 
6. Alternatively, if you would like to rerun the model, you may import the file into Google Colab. 
7. Website functions are covered in the Demo portion. 

<!-- USAGE EXAMPLES -->
## Usage

The main file is:

```Python
        vixm_adaboost_model.ipynb
````
    
which is a Jupyter Notebook with a pre-run code. You can go through it and see code as well as results. 

If you look to reuse the code, and do not have experience on jupyter lab, you can refer [this tutorial.](https://www.dataquest.io/blog/jupyter-notebook-tutorial)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Token Value -->
## Model Specifications
Coming soon. Details are in the model file itself per now. For the VIX first model, you can get a summary in the README file located [here].

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License
<div align="left">
Distributed under the MIT License.
https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact
<div align="left">

Paola Carvajal- (https://www.linkedin.com/in/paolacarvajal/)

  Project Link: [https://github.com/paocarvajal1912](https://github.com/paocarvajal1912)


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Credit to creator of readme template. 
* [README Template](https://github.com/othneildrew/Best-README-Template.git)
    
The first version of this code was made to predict the VIX Index. It was the team lider of the project which results you can check 
[here](https://github.com/paocarvajal1912/vix_predictor) and I work it together in the adaboost moel code with Sangram Singh - @Github - sangramsinghg@yahoo.com, which contributions where invaluable.

<p align="right">(<a href="#top">back to top</a>)</p>

