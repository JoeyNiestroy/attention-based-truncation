# attention-based-truncation
A text input truncation method based on average attention scores from transformer based LLM

### See Study at Bottom ###  

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

# Evaluating the Impact of Attention Truncation on Transformer Models: A Comparative Study

## Abstract
This study explores the effectiveness of attention truncation in transformer-based models, particularly focusing on the bidirectional architecture exemplified by BERT (Bidirectional Encoder Representations from Transformers). We conduct an experiment comparing three approaches: a control model with no truncation, traditional right-side truncation, and an innovative attention-based truncation method. The experiment is conducted using the 20NewsGroups dataset. Our preliminary results demonstrate that attention truncation significantly outperforms traditional right-side truncation and closely approaches the performance of the non-truncated model.

## Introduction
Transformer models, particularly those based on bidirectional architectures like BERT, have revolutionized the field of natural language processing. However, these models often face challenges with lengthy input sequences. This study investigates the potential of attention truncation as a technique to handle such sequences effectively, aiming to maintain high model performance while managing computational constraints.

## Experiment Design

### Data Source
The experiment utilizes the 20NewsGroups dataset, a widely recognized benchmark in text classification tasks. This dataset is publicly available for download on the Hugging Face platform.

### Preprocessing
Approximately 20% of the initial training and development set, which had encoded lengths exceeding 512 tokens, was excluded to maintain consistency with our experimental controls. The dataset was used as-is without further preprocessing to avoid introducing biases. The original test/train split provided in the dataset was retained, with plans to implement 5-fold cross-validation in future iterations of the study.

### Methodology

#### Control Model
The control model, employing a BERT base uncased architecture without any configuration changes, processed data where the tokenized length was 512 or less. This model underwent training for two epochs. The complete hyperparameters are detailed in the accompanying code repository.

#### Traditional Truncation Model
For this model, traditional right-side truncation was employed, discarding 50% of the tokens from each sample. For example: a sample with a tokenized length of 100 was truncated to 50 or a sample with a tokenized length of 50 was truncated to 25. Training parameters were kept consistent with the control model for a fair comparison.

#### Attention Truncation Model
This model initially mirrored the traditional truncation approach during its first epoch. Subsequently, an attention dictionary was created for the full token vocabulary, and the second epoch employed attention-based truncation for token removal. All other training parameters were aligned with those of the control model.

## Results and Discussion

### Model Performance

| Model                  | Description                 | Accuracy (%)  |
|------------------------|-----------------------------|---------------|
| Control Model          | No truncation applied       | 74%           |
| Traditional Truncation | 50% Right-side Truncation   | 66%           |
| Attention Truncation   | Attention-Based Truncation  | 71%           |

The results indicate that attention truncation outperforms traditional right-side truncation by a significant margin, showcasing only a 3% decrease in performance compared to the non-truncated control model. This suggests that attention truncation could be a viable alternative in scenarios where input sequence length exceeds model constraints.

## Conclusion
This study presents a novel approach to managing long input sequences in transformer models through attention truncation. The findings highlight the potential of this method in maintaining model efficacy while addressing the limitations posed by lengthy inputs. Future work will focus on expanding the dataset, employing cross-validation techniques, and further optimizing the attention truncation algorithm for broader applicability in natural language processing tasks.

---

*Note: This experiment is currently ongoing, and results will be updated accordingly. The repository for this experiment will be made available upon completion.*


