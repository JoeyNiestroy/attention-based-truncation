import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np



tf.keras.mixed_precision.set_global_policy('mixed_float16')

training_data = pd.read_csv("Training_Data.csv")
training_data = training_data.dropna()

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

new_tokens = ['<EndofA>',"<EndofQ>", "<Text>", "</Text>", "<code>", "</code>"]
num_added_tokens = tokenizer.add_tokens(new_tokens)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'


# Resize the GPT-2 model's embedding layer to accommodate the new vocabulary size
model.resize_token_embeddings(len(tokenizer))

input_texts = list(training_data["Questions"])[0:20000]
target_texts = list(training_data["Answers"])[0:20000]

for idx,sent in enumerate(input_texts):
    input_texts[idx] = sent+"<EndofQ>"

for idx,sent in enumerate(target_texts):
    target_texts[idx] = sent+"<EndofA>"


# Tokenize the input and target texts
max_length = 1024
tokenizer.padding_side = "left"
input_encodings = tokenizer(input_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='tf')
tokenizer.padding_side = "right"
target_encodings = tokenizer(target_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='tf')


# Convert the input and target encodings to TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices((dict(input_encodings), dict(target_encodings)))

# Define the model training configuration
batch_size = 1
num_epochs = 1
lr = 5e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue= 1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

total_batches = len(dataset) // batch_size

#Intialize the Training Bar
progress_bar = tqdm(total=total_batches * num_epochs, desc="Training Progress")
count = 0

accumulated_gradients = [tf.zeros_like(variable) for variable in model.trainable_variables]
batch_counter = 0
# Define the training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in dataset.shuffle(len(dataset)).batch(batch_size):
        with tf.GradientTape() as tape:
            input_ids = batch[0]['input_ids']
            attention_mask = batch[0]['attention_mask']
            labels = batch[1]['input_ids']
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            current_loss = loss(labels, logits)
        #Handle nan losses
        if np.isnan(current_loss):
            count += 1
            current_loss = 5.0
        else:
            gradients = tape.gradient(current_loss, model.trainable_variables)
        
            # Accumulate the gradients
            accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]
            batch_counter += 1

            if batch_counter == 24:
                # Update weights
                optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                # Reset accumulated gradients and batch counter
                accumulated_gradients = [tf.zeros_like(variable) for variable in model.trainable_variables]
                batch_counter = 0
        
        epoch_loss += current_loss
        num_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": current_loss})
    lr = lr*.8
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue= 1.0)
    name = 'fine_tuned_model_'+str(epoch)
    #model.save_pretrained(name)
    average_loss = epoch_loss / num_batches
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
progress_bar.close()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model_test')
tokenizer.save_pretrained('fine_tuned_pre')