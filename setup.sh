mkdir -p ~/.streamlit/
echo "\
[theme]\n\
primaryColor=\"#f7f6f6\"\n\
backgroundColor=\"#101010\"\n\
secondaryBackgroundColor=\"#ffffff\"\n\
textColor=\"#b9babd\"\n\
\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml