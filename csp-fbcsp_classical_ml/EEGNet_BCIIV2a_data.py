from graphviz import Digraph

# Create a directed graph with improved styling
pipeline = Digraph("EEGNet_Pipeline", format="png")
pipeline.attr(rankdir="LR", size="20,10", ratio="auto",
             pad="0.5", nodesep="0.2", ranksep="0.6")
pipeline.attr("node", shape="box", style="filled,rounded", 
             fontname="Helvetica", fontsize="11", margin="0.15,0.03")
pipeline.attr("edge", fontname="Helvetica", fontsize="10", arrowsize="0.8")

# Softer Color palette (gradient from light blue to darker blue)
blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]

# Main Steps
with pipeline.subgraph(name="cluster_main") as main:
    main.attr(style="filled", color="whitesmoke", bgcolor="whitesmoke")
    
    # Create invisible nodes to force alignment
    main.node("start", style="invis")
    
    # Step 1 - Data Loading and Preprocessing
    with main.subgraph(name="cluster_step1") as step1:
        step1.attr(style="filled", color=blue_shades[0], fillcolor=blue_shades[0])
        step1.node("Step1", "DATA LOADING & PREPROCESSING", fontname="Helvetica-Bold", fontsize="12")
        step1.node("Step1_1", "Load BCI IV 2a data (from npz files)", fontname="Helvetica", fontsize="10")
        step1.node("Step1_2", "Bandpass filter (4-40Hz) using Butterworth", fontname="Helvetica", fontsize="10")
        # Vertical connections
        step1.edge("Step1", "Step1_1", style="invis")
        step1.edge("Step1_1", "Step1_2", style="invis")

    # Step 2 - Epoch Extraction
    with main.subgraph(name="cluster_step2") as step2:
        step2.attr(style="filled", color=blue_shades[1], fillcolor=blue_shades[1])
        step2.node("Step2", "EPOCH EXTRACTION", fontname="Helvetica-Bold", fontsize="12")
        step2.node("Step2_1", "Extract epochs based on event markers", fontname="Helvetica", fontsize="10")
        step2.node("Step2_2", "Create epochs for each class (left, right, rest, etc.)", fontname="Helvetica", fontsize="10")
        # Vertical connections
        step2.edge("Step2", "Step2_1", style="invis")
        step2.edge("Step2_1", "Step2_2", style="invis")

    # Step 3 - Model Training
    with main.subgraph(name="cluster_step3") as step3:
        step3.attr(style="filled", color=blue_shades[2], fillcolor=blue_shades[2])
        step3.node("Step3", "MODEL TRAINING (EEGNet)", fontname="Helvetica-Bold", fontsize="12")
        step3.node("Step3_1", "Initialize EEGNet with 2 classes", fontname="Helvetica", fontsize="10")
        step3.node("Step3_2", "Train using Keras (with Adam optimizer, categorical crossentropy)", fontname="Helvetica", fontsize="10")
        step3.node("Step3_3", "ModelCheckpoint callback (save best model)", fontname="Helvetica", fontsize="10")
        # Vertical connections
        step3.edge("Step3", "Step3_1", style="invis")
        step3.edge("Step3_1", "Step3_2", style="invis")
        step3.edge("Step3_2", "Step3_3", style="invis")

    # Step 4 - Evaluation
    with main.subgraph(name="cluster_step4") as step4:
        step4.attr(style="filled", color=blue_shades[3], fillcolor=blue_shades[3])
        step4.node("Step4", "MODEL EVALUATION", fontname="Helvetica-Bold", fontsize="12")
        step4.node("Step4_1", "Evaluate model on test set", fontname="Helvetica", fontsize="10")
        step4.node("Step4_2", "Accuracy score calculation", fontname="Helvetica", fontsize="10")
        step4.node("Step4_3", "Save detailed and summary results", fontname="Helvetica", fontsize="10")
        # Vertical connections
        step4.edge("Step4", "Step4_1", style="invis")
        step4.edge("Step4_1", "Step4_2", style="invis")
        step4.edge("Step4_2", "Step4_3", style="invis")

    # Force horizontal alignment of main steps
    main.edge("start", "Step1", style="invis")
    main.edge("Step1", "Step2", style="invis")
    main.edge("Step2", "Step3", style="invis")
    main.edge("Step3", "Step4", style="invis")

# Connect main steps with visible edges
pipeline.edge("Step1", "Step2", label="Data Preprocessing", color="#555555", penwidth="1.2")
pipeline.edge("Step2", "Step3", label="Epochs created", color="#555555", penwidth="1.2")
pipeline.edge("Step3", "Step4", label="Model trained", color="#555555", penwidth="1.2")

# Title
pipeline.attr(label="EEGNet Motor Imagery Classification Pipeline", 
              labelloc="t", labeljust="c", fontsize="16", fontname="Helvetica-Bold")

# Render
pipeline.render("EEGNet_Pipeline", view=True, format="pdf", engine="dot")