# Comparison of Machine-Translations Models

<div align="center">
 <p>
    <img style="" src="./logounipi.png" alt="Logo" width="250" >  <br>
  </p>
</div>
<br>
<div align="center">
 <p align="center"><h3>Authors</h3>
    <a href="mailto:g.acciaro@studenti.unipi.it">Gennaro Daniele Acciaro</a>
    ·
    <a href="mailto:@studenti.unipi.it"> - - </a>   
    ·
    <a href="mailto:@studenti.unipi.it"> - - </a>
  </p>
    <p align="center">
    <h3><a href="./report.pdf">Report</a></h3>
  </p>
    <!-- <p align="center">
            <h3><a href="./slides.pdf">Slides</a></h3>
          </p>
        -->
</div>

### Abstract
TODO: add abstract

Master Degree (Artificial Intelligence curriculum)<br>
**HLT** course, Academic Year: 2021/2022<br>

## 🔧 Setup
The files containing the weights of the various models used were not included in this repository because they are very large files.
Therefore, in order for the program to work, it is necessary to download the weights files using the following command.

    ./download_weights.sh


## 🖥 GUI
<div align="center">
 <p>
    <img style="" src="./screenshot.png" >  <br>
  </p>
</div>

### Main Files

|File   |Description   |
|---|---|
| [main_monk.py](./main_monk.py)  | Our best model for MONKs' problems
| [main_cup.py](./main_cup.py) | Our best model for CUP's problem |
| [model_selection_monk.py](./model_selection_cup.py) | The starting point of model selection for MONKs' problems  |
| [model_selection_cup.py](./model_selection_cup.py) |  The starting point of model selection or CUP's problem |
| [model_selection_cup_distributed.py](./validation/distribuited_computing/model_selection_cup_distributed.py) | The starting point of __distribuited__ model selection or CUP's problem  (Note: it requires a database and its initialization) |
| [AIAIAI_ML-CUP21-TS.csv](./AIAIAI_ML-CUP21-TS.csv) | Our Results for the Blind TS |
| [report.pdf](./report.pdf) | Our report |
