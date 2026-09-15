# Threat Conditioning — Context Plan

## Scope
Local rules for the threat_conditioning PsychoPy experiment.

## Goals
Create an experiment which presents frequency-tagged visual stimuli, sends event codes, and ultimately uses imported eeg data to extract SSVEP readouts. 

## Data organization
```
data/                       
  sub-<id>/
    ses-<id>/ 
      rawdata/
        sub-<id>_ses-<id>_run-<id>_eeg.cnt
      metadata/
        session_info_run-<id>.json          # session metadata
        trial_conditions_run-<id>.csv               # trial structure  
        questplus_run-<id>.json             # all condition-specific QUEST+ states
      derivatives/              # analysis outputs (fft, snr, plots)
```

## Rules
- always request permission before applying edits
- do not make assumptions about experimental decisions
- preserve frame-based timing logic 
- do not modify existing data files
- suggest higher-order functions when they reduce redundancy
- suggest opportunities for clean-up and readability
- suggest tests and test vectors that allow for validation of frame-based timing logic; trial condition generation and logging; and event trigger codes

- names of experimental parameters should represent their scientific meaning and use units when applicable
- names of functions should represent their function within the program
- do not use uppercases for naming
- keep parameters at the top of a file or in config files
- keep helper functions at the bottom of the file
- preserve ordering of related variables
- conditionals: changing variable first
- use short, concise comments for non-obvious logic or reasoning

## Documentation
- keep docstrings short and straightforward
