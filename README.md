# attention-based-truncation
A text input truncation method based on average attention scores from transformer based LLM

# Overview and Motivations #
When working on finetuning a GPT-2 model for python Q&A, I ran into two issues. Out-of-Memory errors and poor model preformance on longer prompts. Even pushing the max length as far as I could with batch_sizes of 1 and accumulating gradients there was little improvments. As of right now more advanced truncation algos are unavailable through hugging face. So I decided to write my own truncation function based on the average attention scores assigned to each token by my fine-tuned model. My hypothesis being if I removed tokens based on their relative importance to the model I can retain more information in smaller prompts/sequences and inference would have a smaller memory load or retain more information in longer promts. Since we're not all OpenAI we can't push max input lengths to 16k. Also with the movement towards smaller more comp and memory efficient LLMs (Orca, alpaca, etc) it makes sense to push for that as well in our model inputs.

## Code and Methods ##
The model I used for my attention scores was a GPT-2 small that I fine-tuned off 20k Q&A pairs of samples, for this orginal training my max length was 1024 (largest I could fit into my GPU). It was only trained for 1 epoch, I'll attach code to repo if you're curious about the full hyperparameters. Then I used that model and the same training data to compute the average attention scores for each token in my training vocabulary. The averages may not fully capture the more complex patterns such as token position and surround vocabulary but I beleive it gives a high level view of what's important to the model. This is the code block used to compute these:
![code_block](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/d8371115-c601-4a57-8923-dc533cdb178c)

After getting the attention scores a dictionary is created mapping tokens and their avg scores together. In this case my dictionary contained rouhgly 25k tokens or in my case around 80% of the tokens in my full training set so not too bad. 

Following this I wrote the function to actually truncate encoded sequences. It simply uses the created dictonary to remove tokens based on their relative importance/information. As it stands right now it only works with lists, but I'll extend functionality soon. Below is the code for said function:

![function](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/febbf60b-73c5-498f-b347-0721bc36e620)

There's also no exceptions/error handling but I figure if anybody wants to test it out themseleves they can choose how to best handle them for their purposes

Below is an example of a truncated text using this method:
![Example_trunc](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/cdee5f1a-7c3c-4659-91fb-ce587bdde1ed)

I've attached all my code to this repo. The training data was top 25k python questions and answers I webscaped and processed from StackoverFlow. I would open source dataset, but I did not follow guidelines for proper open source license so if someone is interested in it just let me know. 

GPT.py - The python script to train my QA model

LVIT.ipynb - Jupyter file included from readme, you can import base models to test functions however data is too large to attach 


## Further work ##
I'm currenlty working on testing the method out on a larger training set both in it's deployment during training. As well as testing it's effectiviness during inference by comparing the outputs of tradtional truncation and attention truncation using BERT for a similarty model. Also testing for other downstream tasks (sentiment,classification,etc). I'd also like to explore looking at the attention scores from only the first layers, 1-2, 1-3, etc. Obvisouly past the first layer the embeddings passed forward may not truly 'represent' their original tokens. If anybody with more compute power wants to test it out please let me know how it goes. You can email me at niestroyjoe@gmail.com or reach out where ever. 

# Experiment #
I will create a seperate repo for the experiment when it is complete but it will be here for now. **This experiement is currently in progress. Results will be updated**

## Overview ##
I determined the best transformer models to intially test the effectivness of Attention Trunction would be bidirectional style models like BERT. So base BERT uncased will be the starting model throughout the extent of this section of the experiemnt. No config changes were made to BERT. The tokenizer used was of course the BERT tokenizer

## Experiment Setup

### Dataset
The first dataset used was 20NewsGroups fixed. The full dataset is avilable on hugging face for download

### Preprocessing
Roughly 20% of the intital traninig/dev dataset was removed as it's encoded length was > 512 and so did not fit the expirmental controls (See control model below) 

There was no preprocessing done to the text as the data was already fairly proccessed at download and did not want to introduce any new biases

The original test/train split provided in the dataset was used for the experiment, however when I free time I will set up 5 fold CV 

## Methodology

### Control Model
For the control model all training data was not truncated, or to say all data was <= 512 length when tokenized. The model was trained for 2 epochs, full hyperparameters can be found in code.

### Traditional Truncation Model
Traditional right side truncation was used for this group and 50% of tokens were removed from each sample.

Ex: A sample of tokenized length 100 would be truncated to 50, A sample of length 60 would be truncated to length 30

This model was also trained for 2 epochs with matching hyperparameters to control group

### Attention Truncation Model
In this model the first epoch is the same as the traditional truncation group, however after this I used my code provided above to create an attention dictonary for the full token vocabulary. Then for the second epoch the 50% token removal was done using Attention truncation. 

All other hyperparameters were same as control group

## Model Performance Comparison

| Model                  | Description                 | Accuracy (%)  |
|------------------------|-----------------------------|---------------|
| Control Model          | No truncation applied       | 74%  |
| Traditional Truncation | 50% Right side      | 66%   |
| Attention Truncation       | Attention Based Truncation | 71%   |

As you can clearly see above attention truncatio signficatly outprefroms right side truncation and is only **3%** loss in performance from no truncation at all


