

echo "🚀 Setting up Multimodal AI Chatbot..."

# Step 1: Create a virtual environment
python -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip
pip install --upgrade pip

# Step 3: Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Step 4: Run the chatbot
echo "✅ Setup complete! Starting chatbot..."
python main.py
