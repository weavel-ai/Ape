# BIRD-bench

## Source

[BIRD-bench](https://bird-bench.github.io/)

## Description

**BIRD-bench** is a benchmark for SQL generation tasks. It contains 12,751 question-SQL pairs across 95 databases, covering 37 professional domains.

## Motivation

SQL generation is a common use case for large language models (LLMs), especially for specific databases. To evaluate prompt optimization performance on a more realistic and focused task, we chose a small subset of BIRD-bench using a single database. This allows for testing optimization methods in a practical setting with a manageable dataset.

## Preprocessing

1. From the training split, select samples where `db_id == "works_cycles"` because it is the largest database with the most samples in BIRD-bench.
2. Shuffle the dataset using seed `1`.
3. Select the first 100 samples (`[:100]`) as the training set and the next 100 samples (`[100:200]`) as the validation set.
4. Build the table schema using the following code:

   ```python
   def generate_schema_prompt_sqlite(db_path, num_rows=None):
       full_schema_prompt_list = []
       conn = sqlite3.connect(db_path)
       cursor = conn.cursor()
       cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
       tables = cursor.fetchall()
       schemas = {}
       for table in tables:
           if table == "sqlite_sequence":
               continue
           cursor.execute(
               "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                   table[0]
               )
           )
           create_prompt = cursor.fetchone()[0]
           schemas[table[0]] = create_prompt
           if num_rows:
               cur_table = table[0]
               if cur_table in ["order", "by", "group"]:
                   cur_table = "`{}`".format(cur_table)

               cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
               column_names = [description[0] for description in cursor.description]
               values = cursor.fetchall()
               rows_prompt = nice_look_table(column_names=column_names, values=values)
               verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                   num_rows, cur_table, num_rows, rows_prompt
               )
               schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

       for k, v in schemas.items():
           full_schema_prompt_list.append(v)

       schema_prompt = "\n\n".join(full_schema_prompt_list)

       return schema_prompt
   ```

5. Use the `question` and `schema_prompt` columns as inputs, and format the `sql` column as the output in the following structure: `{"answer": "..."}`.

## Evaluation

We evaluate the results of SQL query using the average of the Exact Match (EM) score and F1 score. The following Python code is used for this evaluation:

```python
def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

def execute_sql(sql, db_path, return_time=False):
    # Connect to the database
    conn = connect_db(db_path)
    start_time = time.time()
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    conn.close()  # Don't forget to close the connection!
    exec_time = time.time() - start_time
    if return_time:
        return exec_time

    return res

# Calculate exact match
def calculate_exact_match(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def calculate_row_match(predicted_row, ground_truth_row):
    """
    Calculate the matching percentage for a single row.

    Args:
    predicted_row (tuple): The predicted row values.
    ground_truth_row (tuple): The actual row values from ground truth.

    Returns:
    float: The match percentage (0 to 1 scale).
    """
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage

# Calculate F1 score
def calculate_f1_score(predicted, ground_truth):
    """
    Calculate the F1 score based on sets of predicted results and ground truth results,
    where each element (tuple) represents a row from the database with multiple columns.

    Args:
    predicted (set of tuples): Predicted results from SQL query.
    ground_truth (set of tuples): Actual results expected (ground truth).

    Returns:
    float: The calculated F1 score.
    """
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # Drop duplicates
    predicted_set = set(predicted) if predicted else set()
    ground_truth_set = set(ground_truth)

    # convert back to list
    predicted = list(predicted_set)
    ground_truth = list(ground_truth_set)

    # Calculate matching scores for each possible pair
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    for i, gt_row in enumerate(ground_truth):
        # rows only in the ground truth results
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(
            pred_row, gt_row
        )
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    # rows only in the predicted results
    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return f1_score
```
