import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
# from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict

class GemmaSelfIE:
    def __init__(self,
                 model,
                 tokenizer,
                 interpretation_prompt = "What is the following? Answer briefly",
                 num_placeholders = 2,
                 max_new_tokens = 50):
        """
        Initialize the SelfIE interpreter with a pre-loaded model and tokenizer.
        
        Args:
            model (PreTrainedModel): The pre-initialized language model
            tokenizer (PreTrainedTokenizer): The corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = self.model.model.layers
        self.position_embedding = self.model.model.rotary_emb
        self.interpretation_prompt = interpretation_prompt
        self.num_placeholders = num_placeholders
        self.max_new_tokens = max_new_tokens  
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_interpretation_prompt(
        self, 
        prompt_sequence: List[str or None]
    ) -> Dict:
        """
        Create an interpretation prompt with placeholder locations.
        
        Args:
            prompt_sequence (List): A list of strings/None to create the prompt
        
        Returns:
            Dict containing the interpretation prompt details
        
        Example:
            Input: ["Please explain the meaning of: ", None]
            Output: A prompt with a specific structure and tracked insertion locations
        """
        interpretation_prompt = ""
        insert_locations = []
        repeat = self.num_placeholders

        for part in prompt_sequence:
            if isinstance(part, str):
                interpretation_prompt += part
            else:
                # Add a placeholder
                insert_start = len(self.tokenizer.encode(interpretation_prompt))
                # interpretation_prompt += "_ "
                interpretation_prompt += repeat*"_ "
                insert_end = len(self.tokenizer.encode(interpretation_prompt))

                # Track insert locations
                for insert_idx in range(insert_start, insert_end):
                    insert_locations.append(insert_idx)
        
        # Tokenize the prompt
        prompt_inputs = self.tokenizer(interpretation_prompt, return_tensors="pt")
        
        return {
            "prompt": interpretation_prompt,
            "inputs": prompt_inputs,
            "insert_locations": insert_locations
        }
    
    def interpret(self, 
                  to_interpret_text:str,
                  layers_to_interpret = [8,10,12],
                  layers_interpreter = [3,4],
                  token_index = -2,
                  interpretation_prefix='interpretation'):
        
        interpretations = []
        layers = self.layers

        interpretation_prompt = self.create_interpretation_prompt([self.interpretation_prompt, None])
        to_interpret_input = self.tokenizer(to_interpret_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            to_interpret_output = self.model(
                **to_interpret_input, 
                output_hidden_states=True
            )
        
        interpret_dict = {}
        for l in layers_to_interpret:
            for k in layers_interpreter:
                
            # Prepare the model inputs for generation
                interpreter_input = interpretation_prompt['inputs'].to(self.model.device)
                interpreter_input_ids = interpreter_input['input_ids']
                interpreter_input_masks = interpreter_input['attention_mask']
                #Retrieve hidden state to interpret in layer l
                hidden_state_to_interpret = to_interpret_output.hidden_states[l][0][token_index]
                generated_tokens = []

                for _ in range(self.max_new_tokens):
                    # Prepare model inputs
                    #Interpret prompt
                    interpreter_inputs = {
                        'input_ids': interpreter_input_ids,
                        'attention_mask': interpreter_input_masks
                    }

                    # Manually modify the hidden states
                    with torch.no_grad():
                        # Forward of interpretation prompt
                        interpreter_outputs = self.model(
                            **interpreter_inputs, 
                            output_hidden_states=True
                        )
                        # Target the layer where the embedding will be replaced
                        interpreter_hidden_states = interpreter_outputs.hidden_states[k].clone()
                        #Modify for every placeholder token
                        for idx in interpretation_prompt['insert_locations']:
                            interpreter_hidden_states[0, idx-1, :] = hidden_state_to_interpret
                        for i in range(k, len(layers)):
                            seq_len = interpreter_hidden_states.shape[1]
                            interpreter_position_ids = torch.arange(seq_len, dtype=torch.long, device=self.model.device)
                            interpreter_position_ids = interpreter_position_ids.unsqueeze(0).expand_as(interpreter_input_masks)
                        # Get position embeddings from the model's embedding layer
                            position_embeddings = self.model.model.rotary_emb(interpreter_hidden_states,interpreter_position_ids)
                            interpreter_hidden_states = layers[i](
                                    interpreter_hidden_states,
                                    position_ids=interpreter_position_ids,
                                    position_embeddings=position_embeddings
                                )[0]
                        interpreter_hidden_states = self.model.model.norm(interpreter_hidden_states)
                        logits = self.model.lm_head(interpreter_hidden_states[:, -1, :])
                    
                    # Sample the next token
                    next_token = torch.argmax(logits, dim=-1)
                    # print(next_token)
                    
                    # Append to generated tokens
                    generated_tokens.append(next_token.item())
                    
                    # # Update current input
                    interpreter_input_ids = torch.cat([interpreter_input_ids, next_token.unsqueeze(0)], dim=-1)
                    interpreter_input_masks = torch.cat([interpreter_input_masks, torch.ones(1, 1, device=interpreter_input_masks.device)], dim=-1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                # Decode the generated tokens
                interpretation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                interpret_dict[f'{interpretation_prefix}_{l}.{k}'] = interpretation
                # interpretations.append(interpret_dict)
        return(interpret_dict)
        # return(interpretations)
    
    def interpret_with_LIG(self, 
                  to_interpret_text:str,
                  layers_to_interpret = [8,10,12],
                  layers_interpreter = [3,4],
                  token_index = -2,
                  n_steps=30,
                  intensity=1,
                  isolate=False):
        
        def generate_logits_prediction(input):
            output = self.model(input.to(self.model.device)).logits[:, -1, :]
            return(output)
        
        layers = self.layers

        interpretation_prompt = self.create_interpretation_prompt([self.interpretation_prompt, None])
        to_interpret_input = self.tokenizer(to_interpret_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            to_interpret_output = self.model(
                **to_interpret_input, 
                output_hidden_states=True
            )
        
        interpret_dict = {}
        for l in layers_to_interpret:
            for k in layers_interpreter:
                
            # Prepare the model inputs for generation
                interpreter_input = interpretation_prompt['inputs'].to(self.model.device)
                interpreter_input_ids = interpreter_input['input_ids']
                interpreter_input_masks = interpreter_input['attention_mask']
                
                #Retrieve hidden state fo interpret in layer l
                # hidden_state_to_interpret = to_interpret_output.hidden_states[l][0][token_index]
                layer = self.model.model.layers[l]
                lig = LayerIntegratedGradients(generate_logits_prediction, layer)
                input = self.tokenizer(to_interpret_text, return_tensors="pt")['input_ids'].to(self.model.device)
                baseline = torch.clone(input, memory_format=torch.preserve_format).to(self.model.device)
                baseline[0][token_index] = 0
                idx_max = torch.argmax(self.model(input).logits[:, -1, :])
                attribution = lig.attribute(input, target=idx_max, return_convergence_delta=False, n_steps=n_steps, baselines=baseline)
                ##Adding in the direction of the gradients
                if isolate == False:
                    hidden_state_to_interpret = intensity * attribution[0,token_index,:] + to_interpret_output.hidden_states[l][0][token_index]
                else:
                    hidden_state_to_interpret = intensity * attribution[0,token_index,:] 

                generated_tokens = []

                for _ in range(self.max_new_tokens):
                    # Prepare model inputs
                    #Interpret prompt
                    interpreter_inputs = {
                        'input_ids': interpreter_input_ids,
                        'attention_mask': interpreter_input_masks
                    }

                    # Manually modify the hidden states
                    with torch.no_grad():
                        # Forward of interpretation prompt
                        interpreter_outputs = self.model(
                            **interpreter_inputs, 
                            output_hidden_states=True
                        )
                        # Target the layer where the embedding will be replaced
                        interpreter_hidden_states = interpreter_outputs.hidden_states[k].clone()
                        #Modify for every placeholder token
                        for idx in interpretation_prompt['insert_locations']:
                            interpreter_hidden_states[0, idx-1, :] = hidden_state_to_interpret
                        for i in range(k, len(layers)):
                            seq_len = interpreter_hidden_states.shape[1]
                            interpreter_position_ids = torch.arange(seq_len, dtype=torch.long, device=self.model.device)
                            interpreter_position_ids = interpreter_position_ids.unsqueeze(0).expand_as(interpreter_input_masks)
                        # Get position embeddings from the model's embedding layer
                            position_embeddings = self.model.model.rotary_emb(interpreter_hidden_states,interpreter_position_ids)
                            interpreter_hidden_states = layers[i](
                                    interpreter_hidden_states,
                                    position_ids=interpreter_position_ids,
                                    position_embeddings=position_embeddings
                                )[0]
                        interpreter_hidden_states = self.model.model.norm(interpreter_hidden_states)
                        logits = self.model.lm_head(interpreter_hidden_states[:, -1, :])
                    
                    # Sample the next token
                    next_token = torch.argmax(logits, dim=-1)
                    # print(next_token)
                    
                    # Append to generated tokens
                    generated_tokens.append(next_token.item())
                    
                    # # Update current input
                    interpreter_input_ids = torch.cat([interpreter_input_ids, next_token.unsqueeze(0)], dim=-1)
                    interpreter_input_masks = torch.cat([interpreter_input_masks, torch.ones(1, 1, device=interpreter_input_masks.device)], dim=-1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                # Decode the generated tokens
                interpretation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                interpret_dict[f'{l}.{k}'] = interpretation
                # interpretations.append(interpret_dict)
        return(interpret_dict)
