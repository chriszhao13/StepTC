
# StepTC: Stepwise Triangle Counting on GPU with Two Efficient Set Intersection methods
Welcome to the StepTC project!

We propose a novel stepwise triangle counting algorithm (StepTC). StepTC selects appropriate set intersection methods for each counting step to ensure optimal efficiency on a global scale. It is noteworthy that StepTC outperforms state-of-the-art solutions, delivering impressive speedups ranging from 1.4× to 22.4× on various datasets.

## Dependencies
```
    CUDA Toolkit == 11.7 
    g++ == 9.4.0
    vmtouch == latest
```


## Build
```
    git clone https://github.com/anonymizednickname/StepTC.git
    cd StepTC
    make
    sh run.sh
```

## Performance

| Graphs | StepTC Time(ms) |
| :---: | :---: |
| liveJournal  | 23    |
| orkut        | 102   |
| twitter20    | 767   |
| twitter_rv   | 2403  |
| friendster   | 2224  |
| s20.kron     | 26    |
| s21.kron     | 59    |
| s22.kron     | 146   |
| s23.kron     | 332   |
| s24.kron     | 781   |
| s25.kron     | 1854  |

## Datasets
    Available in 