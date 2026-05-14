#!/bin/bash
cd "$(dirname "$0")"

if [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
elif [ -x "venv/bin/python3" ]; then
    PY="venv/bin/python3"
else
    echo "Sanal ortam bulunamadi, sistem python3 kullaniliyor."
    PY="python3"
fi

echo "=========================================="
echo "  Beyin BT Siniflandirma - Streamlit"
echo "=========================================="
echo "Python: $PY"
echo ""

"$PY" -m streamlit run src/arayuz/arayuz_app.py
