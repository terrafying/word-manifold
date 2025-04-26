import pytest
from word_manifold.manifold.agent_explorer import AgentManifoldExplorer
from word_manifold.manifold.timeseries import SemanticTimeSeriesAnalyzer
from word_manifold.embeddings import get_embeddings
from llama_stack_client import Agent

def test_agent_semantic_exploration():
    """Test agent-based semantic exploration."""
    # Create agent
    agent = Agent(
        model="meta-llama/Llama-3-70b-chat",
        instructions="You are a semantic space explorer that helps analyze relationships between words."
    )
    
    # Create session
    session_id = agent.create_session(session_name="test")
    
    # Create explorer
    explorer = AgentManifoldExplorer(agent)
    
    # Test words
    words = ["cat", "dog", "bird", "fish", "mammal", "reptile"]
    
    # Explore space
    result = explorer.explore_semantic_space(words, session_id)
    
    # Verify result structure
    assert "satisfiable" in result
    assert "model" in result
    assert "constraints" in result
    assert "variables" in result
    
    # Test semantic path finding
    path = explorer.find_semantic_path("cat", "dog", session_id)
    assert isinstance(path, list)
    assert len(path) > 0

def test_timeseries_analysis():
    """Test time series analysis of semantic evolution."""
    # Sample passage
    passage = "The cat chased the mouse through the garden. The mouse escaped into a hole. The cat waited patiently."
    
    # Get embeddings
    words = passage.split()
    embeddings = get_embeddings(words)
    
    # Create analyzer
    analyzer = SemanticTimeSeriesAnalyzer(window_size=5, stride=1)
    
    # Analyze passage
    result = analyzer.analyze_passage(passage, embeddings)
    
    # Verify result structure
    assert "patterns" in result
    assert "shifts" in result
    assert "graph_metrics" in result
    assert "reduced_dimensions" in result
    
    # Verify pattern analysis
    patterns = result["patterns"]
    assert "velocity_stats" in patterns
    assert "peaks" in patterns
    assert "periodicity" in patterns
    
    # Verify shift detection
    shifts = result["shifts"]
    assert isinstance(shifts, list)
    for shift in shifts:
        assert "position" in shift
        assert "magnitude" in shift
        assert "direction" in shift
        
    # Verify graph metrics
    metrics = result["graph_metrics"]
    assert "density" in metrics
    assert "average_clustering" in metrics
    assert "average_shortest_path" in metrics
    assert "diameter" in metrics
    assert "num_communities" in metrics

def test_combined_analysis():
    """Test combined agent and time series analysis."""
    # Create agent
    agent = Agent(
        model="meta-llama/Llama-3-70b-chat",
        instructions="You are a semantic passage analyzer that helps understand the evolution of meaning in text."
    )
    
    # Create session
    session_id = agent.create_session(session_name="test")
    
    # Create explorer
    explorer = AgentManifoldExplorer(agent)
    
    # Sample passage
    passage = "The cat chased the mouse through the garden. The mouse escaped into a hole. The cat waited patiently."
    
    # Get embeddings
    words = passage.split()
    embeddings = get_embeddings(words)
    
    # Create analyzer
    analyzer = SemanticTimeSeriesAnalyzer(window_size=5, stride=1)
    
    # Analyze passage
    timeseries_result = analyzer.analyze_passage(passage, embeddings)
    
    # Get agent's analysis
    turn_response = agent.create_turn(
        session_id=session_id,
        messages=[{
            "role": "user",
            "content": f"Analyze the semantic evolution in this passage: {passage}. "
                      f"Identify key semantic shifts, patterns, and relationships. "
                      f"Return the analysis as a JSON object with 'shifts', 'patterns', and 'relationships' fields."
        }]
    )
    
    # Process agent response
    agent_result = None
    for log in AgentEventLogger().log(turn_response):
        if log.type == "output":
            try:
                agent_result = json.loads(log.content)
            except json.JSONDecodeError:
                continue
                
    # Verify both analyses
    assert timeseries_result is not None
    assert agent_result is not None
    
    # Compare key aspects
    assert len(timeseries_result["shifts"]) > 0
    assert "shifts" in agent_result
    assert "patterns" in agent_result
    assert "relationships" in agent_result 