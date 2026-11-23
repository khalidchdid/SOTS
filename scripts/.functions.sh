#!/bin/bash
## Use variables defined in before scripts to setup project based on templates

source scripts/.before-script.sh

# Specific function for sed-able files
maybe_replace_logs_template_values(){
    target=$1
    if [ $target == $LOGS_CONFIG_FILE ]; then
        sed -i "s|__LOG_FOLDER__|$LOGS_FOLDER|g" $target
        sed -i "s|__LOG_FILE__|$LOGS_FILE|g" $target
    fi
}

maybe_replace_dotenv_template_values(){
    target=$1
    if [ $DO_USE_AZURE_OPENAI == 1 ] && [ $target == $DOTENV_FILE ]; then
        echo === Enter Azure OpenAI API data ===
        read -p "Azure OpenAI API Endpoint (default to NONE): " AZURE_OPENAI_API_ENDPOINT
        AZURE_OPENAI_API_ENDPOINT=${AZURE_OPENAI_API_ENDPOINT:-NONE}

        read -p "Azure OpenAI API Key (default to NONE): " AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-NONE}

        read -p "Azure OpenAI API Version (default to NONE): " AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-NONE}

        read -p "Azure document intelligence endpoint (default to NONE): " AZURE_DOCINTELLIGENCE_ENDPOINT
        AZURE_DOCINTELLIGENCE_ENDPOINT=${AZURE_DOCINTELLIGENCE_ENDPOINT:-NONE}

        read -p "Azure document intelligence API key (default to NONE): " AZURE_DOCINTELLIGENCE_API_KEY
        AZURE_DOCINTELLIGENCE_API_KEY=${AZURE_DOCINTELLIGENCE_API_KEY:-NONE}

        read -p "Ollama endpoint (default to NONE): " OLLAMA_API_ENDPOINT
        OLLAMA_API_ENDPOINT=${OLLAMA_API_ENDPOINT:-NONE}

        read -p "Ollama API key (default to NONE): " OLLAMA_API_KEY
        OLLAMA_API_KEY=${OLLAMA_API_KEY:-NONE}

        read -p "Ollama API port (default to NONE): " OLLAMA_API_PORT
        OLLAMA_API_PORT=${OLLAMA_API_PORT:-NONE}

        echo replacing api data in $target...
        sed -i "s|__AZURE_OPENAI_API_ENDPOINT__|${AZURE_OPENAI_API_ENDPOINT}|g" $target;
        sed -i "s|__AZURE_OPENAI_API_KEY__|${AZURE_OPENAI_API_KEY}|g" $target;
        sed -i "s|__AZURE_OPENAI_API_VERSION__|${AZURE_OPENAI_API_VERSION}|g" $target;
        sed -i "s|__OPENAI_API_KEY__|${AZURE_OPENAI_API_KEY}|g" $target;
        sed -i "s|__AZURE_DOCINTELLIGENCE_ENDPOINT__|${AZURE_DOCINTELLIGENCE_ENDPOINT}|g" $target;
        sed -i "s|__AZURE_DOCINTELLIGENCE_API_KEY__|${AZURE_DOCINTELLIGENCE_API_KEY}|g" $target;
        sed -i "s|__OLLAMA_API_ENDPOINT__|${OLLAMA_API_ENDPOINT}|g" $target;
        sed -i "s|__OLLAMA_API_KEY__|${OLLAMA_API_KEY}|g" $target;
        sed -i "s|__OLLAMA_API_PORT__|${OLLAMA_API_PORT}|g" $target;
    fi
}

maybe_replace_pyproject_template_values(){
    target=$1
    if [ $target == $PYPROJECT_FILE ]; then
        sed -i "s|__PROJECT_NAME__|$PROJECT_NAME|g" $target
        sed -i "s|__VERSION__|$VERSION|g" $target
        sed -i "s|__PYTHON_VERSION__|$PYTHON_VERSION|g" $target
        sed -i "s|__APP_FILE_MODULE__|$APP_FILE_MODULE|g" $target
        sed -i "s|__SRC_DIR_PKG__|$SRC_DIR_PKG|g" $target
        sed -i "s|__FIRST_NAME__|$FIRST_NAME|g" $target
        sed -i "s|__SURNAME__|$SURNAME|g" $target
        sed -i "s|__EMAIL__|$EMAIL|g" $target
    fi
}

maybe_replace_python_version_template_values(){
    target=$1
    if [ $target == $PYTHON_VERSION_FILE ]; then
        sed -i "s|__PYTHON_VERSION__|$PYTHON_VERSION|g" $target
    fi
}

maybe_replace_pyproject_setup_file_template_values(){
    target=$1
    if [ $target == $PYPROJECT_SETUP_FILE ]; then
        sed -i "s|__BUILD_LIB_DIR__|$BUILD_LIB_DIR|g" $target
    fi
}

maybe_replace_readme_template_values(){
    target=$1
    if [ $target == $README_FILE ]; then
        sed -i "s|__PROJECT_NAME__|$PROJECT_NAME|g" $target
        sed -i "s|__VERSION__|$VERSION|g" $target
    fi
}

maybe_replace_app_template_values(){
    target=$1
    if [ $target == $APP_FILE ]; then
        sed -i "s|__LOGS_CONFIG_FILE__|$LOGS_CONFIG_FILE|g" $target
    fi
}

maybe_replace_license_template_values(){
    target=$1
    if [ $target == $LICENSE_FILE ]; then
        sed -i "s|__PROJECT_NAME__|$PROJECT_NAME|g" $target
    fi
}


# Generic function to add a .gitkeep in a specific folder
add_gitkeep_in_folder(){
    folder=$1
    touch $folder/.gitkeep
    echo \!.gitkeep > $folder/.gitignore
}

add_init_in_folder(){
    folder=$1
    touch $folder/__init__.py
}

# Generic mkdir function
make_a_dir(){
    folder=$1
    if [ ! -d $folder ]; then
        echo creating $folder folder
        mkdir -p $folder;
    else
        echo folder $folder already exist, not recreating it...
    fi
}

# Generic copy template function
copy_template_to_target(){
    template=$1
    target=$2
    if [ -f $target ]; then
        echo $target exists, not copying it from template....;
    else
        echo copying $target from template $template...;
        cp $template $target
        maybe_replace_dotenv_template_values $target
        maybe_replace_logs_template_values $target
        maybe_replace_pyproject_template_values $target
        maybe_replace_pyproject_setup_file_template_values $target
        maybe_replace_python_version_template_values $target
        maybe_replace_readme_template_values $target
        maybe_replace_app_template_values $target
        maybe_replace_license_template_values $target
        
    fi
}