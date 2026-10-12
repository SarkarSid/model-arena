#!/bin/bash
# Azure Web App startup script
# Set in App Service → Configuration → General settings → Startup Command:
#   bash startup.sh
python -m streamlit run app.py \
  --server.port "${PORT:-8000}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection true
