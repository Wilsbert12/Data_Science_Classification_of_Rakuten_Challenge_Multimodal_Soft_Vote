# Replace lines 93-103 in your 6_Prediction.py with this:

# Load ensemble model with lazy loading
@st.cache_resource
def load_ensemble():
    """Load the ensemble classifier with caching - only when needed"""
    classifier = RakutenEnsembleClassifier()
    classifier.load_models()
    return classifier

def get_classifier():
    """Get classifier with lazy loading and progress indicator"""
    if 'classifier' not in st.session_state:
        with st.spinner("🤖 Loading ensemble models... (this may take up to 30 seconds)"):
            try:
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                st.write("Loading SVM + TF-IDF models...")
                
                progress_bar.progress(50)
                st.write("Loading BERT model...")
                
                progress_bar.progress(75)
                st.write("Loading VGG16 model...")
                
                classifier = load_ensemble()
                
                progress_bar.progress(100)
                st.success("✅ Ensemble models loaded successfully!")
                st.session_state.classifier = classifier
                
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                return None
                
    return st.session_state.classifier

# Remove the old loading code (lines 98-103) and replace the prediction buttons with:

# In the sample prediction section, replace the prediction button:
if st.button("🔮 Predict Category", key="predict_sample"):
    classifier = get_classifier()
    if classifier is None:
        st.stop()
        
    with st.spinner("Running ensemble prediction..."):
        # Prepare text
        combined_text = f"{current_sample['title']} {current_sample['description']}"
        cleaned_text = text_cleaner(combined_text)
        
        # Make prediction
        result = classifier.predict_single(
            cleaned_text, 
            current_sample['image_path'], 
            return_details=True
        )
        
        st.markdown("---")
        display_prediction_results(result, category_mapping)

# In the user upload section, replace the prediction logic:
if predict_user:
    if not user_title and not user_description:
        st.warning("⚠️ Please enter at least a title or description")
    elif not uploaded_image:
        st.warning("⚠️ Please upload an image")
    else:
        classifier = get_classifier()
        if classifier is None:
            st.stop()
            
        with st.spinner("Processing your product..."):
            try:
                # Prepare text
                combined_text = f"{user_title} {user_description}".strip()
                cleaned_text = text_cleaner(combined_text)
                
                # Load image
                image = Image.open(uploaded_image).convert("RGB")
                
                # Make prediction
                result = classifier.predict_single(
                    cleaned_text, 
                    image, 
                    return_details=True
                )
                
                st.markdown("---")
                
                # Display input summary
                st.markdown("### 📝 Your Input")
                input_col1, input_col2 = st.columns([1, 2])
                
                with input_col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with input_col2:
                    st.write(f"**Title:** {user_title}")
                    st.write(f"**Description:** {user_description}")
                    st.write(f"**Processed Text:** {cleaned_text[:150]}...")
                
                st.markdown("---")
                display_prediction_results(result, category_mapping)
                
            except Exception as e:
                st.error(f"Error processing your input: {e}")

# In the performance info section, replace with:
with st.expander("📊 Model Performance Information"):
    classifier = st.session_state.get('classifier')
    if classifier:
        try:
            perf_info = classifier.get_performance_info()
            
            st.markdown("### 🏆 Ensemble Performance")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("F1-Score (Weighted)", f"{perf_info['model_info']['ensemble_f1_weighted']:.4f}")
            with perf_col2:
                st.metric("F1-Score (Macro)", f"{perf_info['model_info']['ensemble_f1_macro']:.4f}")
            with perf_col3:
                st.metric("vs Benchmark", perf_info['model_info']['performance_vs_benchmark'])
                
            st.markdown("### ⚙️ Configuration")
            st.json(perf_info['configuration'])
            
        except Exception as e:
            st.error(f"Could not load performance info: {e}")
    else:
        st.info("Load models first by making a prediction to see performance information.")