from flask import Flask, request, render_template
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

app = Flask(__name__)

last_transactions = []
last_support = None
last_confidence = None

def parse_transactions(file):
    filename = file.filename
    if filename.endswith('.csv'):
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        return [line.strip().split(',') for line in lines if line.strip()]
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
        return df.dropna().astype(str).values.tolist()
    else:
        raise ValueError("Unsupported file type")

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_transactions, last_support, last_confidence

    rules_table = ''
    preview_table = ''
    message = ''
    show_form = False

    min_support = request.form.get('min_support', last_support or 0.01)
    min_confidence = request.form.get('min_confidence', last_confidence or 0.3)

    if request.method == 'POST':
        file = request.files.get('file')

        # Upload a new file
        if file and file.filename:
            try:
                last_transactions = parse_transactions(file)
                df_preview = pd.DataFrame(last_transactions[:5])
                preview_table = df_preview.to_html(classes='table table-bordered', header=False)
                show_form = True
                rules_table = ''
                message = ''
            except Exception as e:
                message = f"❌ Error reading file: {e}"
                last_transactions = []

        # Generate rules using last file
        elif last_transactions and min_support and min_confidence:
            try:
                df_preview = pd.DataFrame(last_transactions[:5])
                preview_table = df_preview.to_html(classes='table table-bordered', header=False)
                show_form = True

                min_support = float(min_support)
                min_confidence = float(min_confidence)

                last_support = min_support
                last_confidence = min_confidence

                te = TransactionEncoder()
                te_ary = te.fit_transform(last_transactions)
                df_trans = pd.DataFrame(te_ary, columns=te.columns_)

                frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True)
                if frequent_itemsets.empty:
                    message = "No frequent itemsets found. Try lowering support."
                else:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                    if rules.empty:
                        message = "No rules found. Try lowering confidence."
                    else:
                        rules_table = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']] \
                            .sort_values(by='lift', ascending=False) \
                            .to_html(classes='table table-bordered')
                        message = f"📌 Showing rules for Support = {min_support}, Confidence = {min_confidence}"
            except Exception as e:
                message = f"❌ Error generating rules: {e}"

    return render_template('index.html',
                           preview=preview_table,
                           rules=rules_table,
                           message=message,
                           show_form=show_form,
                           min_support=min_support,
                           min_confidence=min_confidence)


if __name__=="__main__":
    app.run(debug=True)