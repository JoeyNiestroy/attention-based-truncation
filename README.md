# attention-based-truncation
A text input truncation method based on average attention scores from transformer based LLM

# Overview and Motivations #
When working on finetuning a GPT-2 model for python Q&A, I ran into two issues. Out-of-Memory errors and poor model preformance on longer prompts. Even pushing the max length as far as I could with batch_sizes of 1 and accumulating gradients there was little improvments. As of right now more advanced truncation algos are unavilable through hugging face. So I decided to write my own truncation function based on the average attention scores assigned to each token by my fine-tuned model. My hypothesis being if I removed tokens based on their relative importance to the model I can retain more information in smaller prompts/sequences and countinue fine-tuning with larger batches(thus faster) and inference would have a smaller memory load. Since we're not all OpenAI we can't push max input lengths to 16k. Also with the movement towards smaller more comp and memory efficient LLMs (Orca, alpaca, etc) it makes sense to push for that as well in our model inputs.

## Code and Methods ##
The model I used for my attention scores was a GPT-2 small that I fine-tuned off 20k Q&A pairs of samples, for this orginal training my max length was 1024 (largest I could fit into my GPU). It was only trained for 1 epoch, I'll attach code to repo if you're curious about the full hyperparameters. Then I used that model and the same training data to compute the average attention scores for each token in my training vocabulary. The averages may not fully capture the more complex patterns such as token position and surround vocabulary but I beleive it gives a high level view of what's important to the model. This is the code block used to compute these:
![code_block](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/d8371115-c601-4a57-8923-dc533cdb178c)

After getting the attention scores a dictionary is created mapping tokens and their avg scores together. In this case my dictionary contained rouhgly 25k tokens or in my case around 80% of the tokens in my full training set so not too bad. 

Following this I wrote the function to actually truncate encoded sequences. It simply uses the created dictonary to remove tokens based on their relative importance/information. As it stands right now it only works with lists, but I'll extend functionality soon. Below is the code for said function:

![function](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/febbf60b-73c5-498f-b347-0721bc36e620)

There's also no exceptions/error handling but I figure if anybody wants to test it out themseleves they can choose how to best handle them for their purposes

Below is an example of a truncated text using this method:
![Example_trunc](https://github.com/JoeyNiestroy/attention-based-truncation/assets/106636917/cdee5f1a-7c3c-4659-91fb-ce587bdde1ed)

I've attached all my code to this repo

GPT.py - The python script to train my QA model

LVIT.ipynb - Jupyter file included from readme, you can import base models to test functions however data is too large to attach


## Further work ##
I'm currenlty working on testing the method out on a larger training set both in it's deployment during training as well as just inference. If anybody with more compute power wants to test it out please let me know how it goes. You can email me at niestroyjoe@gmail.com or reach out where ever. 
