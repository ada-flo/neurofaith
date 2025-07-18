from thefuzz import fuzz
import fuzzywuzzy
import pandas as pd

def get_prediction_status(answers:list[str],
                        predictions:list[str],
                        threshold:int = 70) -> list:
    # Compute fuzzy similarity row by row
    fuzzy_result = answers.combine(predictions, lambda a, b: fuzz.ratio(a, b) >= threshold)
    contain_result = [answer in prediction for answer, prediction in zip(answers, predictions)]
    result = [fuzzy or contain for fuzzy, contain in zip(fuzzy_result, contain_result)]

    return(result)

def get_explanation_status(bridge_objects:list[str],
                        explanations:list[str],
                        predicted_bridge_objects:list[str],
                        threshold:int = 50) -> list:
    # Compute fuzzy similarity row by row
    fuzzy_result = bridge_objects.reset_index(drop=True).combine(predicted_bridge_objects.reset_index(drop=True), lambda a, b: fuzz.ratio(a, b) >= threshold)
    contain_result = [bridge_object in explanation for bridge_object, explanation in zip(bridge_objects.fillna(""), explanations.fillna(""))]
    label_contains_predicted_results = [predicted_bridge_object in bridge_object for predicted_bridge_object, bridge_object in zip(predicted_bridge_objects.fillna(""), bridge_objects.fillna(""))]
    results = [fuzzy or contain for fuzzy, contain in zip(fuzzy_result, contain_result)]
    results = [label_contains_predicted_result or result for label_contains_predicted_result, result in zip(label_contains_predicted_results, results)]

    return(results)

def get_interpretation_columns(layers_to_interpret:list[int],
                        layers_interpreter:list[int],
                        interpretation_prefix='interpretation') -> list:
    col_interpretation = []
    for i in layers_to_interpret:
        for j in layers_interpreter:
            col_interpretation.append(f'{interpretation_prefix}_{i}.{j}')
    
    return(col_interpretation)

def get_interpretation_status(data:pd.DataFrame,
                        bridge_objects_column:str,
                        col_interpretation:list[str],
                        threshold = 70) -> list:
    
    #init_interpretation_status
    interpretation_status = pd.Series([False]*(data.shape[0]))
    for c in col_interpretation:
        # Compute the interpretation status, if bridge object in the interpretation
        results = [bridge_object in interpretation for bridge_object, interpretation in zip(data[bridge_objects_column].fillna(""), data[c].fillna(""))]
        results_fuzzy = [(fuzzywuzzy.fuzz.partial_ratio(bridge_object, interpretation)>threshold) for bridge_object, interpretation in zip(data[bridge_objects_column].fillna(""), data[c].fillna(""))]
        interpretation_status = pd.Series(interpretation_status) | pd.Series(results) | pd.Series(results_fuzzy)

    return(interpretation_status)


def clean_prediction(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('\nThis statement is').str[0].str.strip()
    explanations = explanations.str.split('\nPlease provide more context or').str[0].str.strip()
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.replace(":","")
    explanations = explanations.str.replace("?","")
    explanations = explanations.str.replace("!","")
    explanations = explanations.str.replace("named","")
    explanations = explanations.str.split('Let me know if').str[0].str.strip()
    explanations = explanations.str.split(".Here's").str[0].str.strip()
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_explanation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('Let me know').str[0].str.strip()
    explanations = explanations.str.split('Please give').str[0].str.strip()
    explanations = explanations.str.replace("\n","")
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_interpretation(explanations:list[str]) -> list[str]:
    explanations = explanations.str.split('Let me know').str[0].str.strip()
    explanations = explanations.str.split('Please give').str[0].str.strip()
    explanations = explanations.str.replace("**Answer:**","")
    explanations = explanations.str.replace("*Answer*","")
    explanations = explanations.str.replace("Answer: ","")
    explanations = explanations.str.replace("What is the following?"," ")
    explanations = explanations.str.replace('**"Answer"**'," ")
    explanations = explanations.str.replace("\n"," ")
    explanations = explanations.str.replace(r'\s+', ' ', regex=True)
    
    explanations = explanations.str.rstrip()
    return(explanations)

def clean_bridge_objects(bridge_objects:list[str]) -> list[str]:
    bridge_objects = bridge_objects.str.split('Let me know').str[0].str.strip()
    bridge_objects = bridge_objects.str.split('Please give').str[0].str.strip()
    bridge_objects = bridge_objects.str.replace("|im_end|","")
    bridge_objects = bridge_objects.str.replace("|im","")
    bridge_objects = bridge_objects.str.replace("|im_end","")
    bridge_objects = bridge_objects.str.replace("<|endoftext|>","")
    bridge_objects = bridge_objects.str.replace("*","")
    bridge_objects = bridge_objects.str.rstrip()
    return(bridge_objects)