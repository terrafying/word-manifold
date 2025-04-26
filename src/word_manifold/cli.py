import click
from typing import List
import json
from .manifold.explorer import ManifoldExplorer
from .manifold.state_space import StateSpaceAnalyzer
from .manifold.agent_explorer import AgentManifoldExplorer
from .manifold.timeseries import SemanticTimeSeriesAnalyzer
from .embeddings import get_embeddings, save_embeddings, load_embeddings
from llama_stack_client import Agent

@click.group()
def cli():
    """Word Manifold CLI for exploring semantic spaces."""
    pass

@cli.command()
@click.argument('words', nargs=-1)
@click.option('--model', default='all-MiniLM-L6-v2', help='Model to use for embeddings')
@click.option('--output', '-o', help='Output file for embeddings')
@click.option('--visualize/--no-visualize', default=True, help='Show visualization')
def explore(words: List[str], model: str, output: str, visualize: bool):
    """Explore the semantic manifold for a set of words."""
    # Generate embeddings
    embeddings = get_embeddings(words, model)
    
    # Save if output specified
    if output:
        save_embeddings(embeddings, output)
    
    # Create explorer
    explorer = ManifoldExplorer(embeddings)
    
    # Analyze structure
    structure = explorer.analyze_structure()
    click.echo("Manifold Structure:")
    click.echo(json.dumps(structure, indent=2))
    
    # Visualize if requested
    if visualize:
        explorer.visualize()

@cli.command()
@click.argument('word1')
@click.argument('word2')
@click.option('--model', default='all-MiniLM-L6-v2', help='Model to use for embeddings')
def path(word1: str, word2: str, model: str):
    """Find the semantic path between two words."""
    # Generate embeddings for both words
    embeddings = get_embeddings([word1, word2], model)
    
    # Create explorer
    explorer = ManifoldExplorer(embeddings)
    
    # Find path
    path = explorer.get_geodesic(word1, word2)
    
    if path:
        click.echo(f"Path from {word1} to {word2}:")
        click.echo(" -> ".join(path))
    else:
        click.echo(f"No path found between {word1} and {word2}")

@cli.command()
@click.argument('constraints', nargs=-1)
def verify(constraints: List[str]):
    """Verify properties in the state space."""
    analyzer = StateSpaceAnalyzer()
    
    # Parse and add constraints
    for constraint in constraints:
        try:
            # Simple constraint format: word1:word2:relation
            word1, word2, relation = constraint.split(':')
            analyzer.add_word_constraint(word1, word2, relation)
        except ValueError:
            click.echo(f"Invalid constraint format: {constraint}")
            return
    
    # Analyze
    result = analyzer.analyze()
    
    if result["satisfiable"]:
        click.echo("Constraints are satisfiable")
        click.echo("Model:")
        click.echo(json.dumps(result["model"], indent=2))
    else:
        click.echo("Constraints are not satisfiable")

@cli.command()
@click.argument('words', nargs=-1)
@click.option('--model', default='meta-llama/Llama-3-70b-chat', help='Llama model to use')
@click.option('--session', default='default', help='Session name')
def agent_explore(words: List[str], model: str, session: str):
    """Explore semantic space using Llama Stack agent."""
    # Create agent
    agent = Agent(
        model=model,
        instructions="You are a semantic space explorer that helps analyze relationships between words."
    )
    
    # Create session
    session_id = agent.create_session(session_name=session)
    
    # Create explorer
    explorer = AgentManifoldExplorer(agent)
    
    # Explore space
    result = explorer.explore_semantic_space(words, session_id)
    
    click.echo("Semantic Space Analysis:")
    click.echo(json.dumps(result, indent=2))

@cli.command()
@click.argument('word1')
@click.argument('word2')
@click.option('--model', default='meta-llama/Llama-3-70b-chat', help='Llama model to use')
@click.option('--session', default='default', help='Session name')
def agent_path(word1: str, word2: str, model: str, session: str):
    """Find semantic path using Llama Stack agent."""
    # Create agent
    agent = Agent(
        model=model,
        instructions="You are a semantic path finder that helps discover connections between words."
    )
    
    # Create session
    session_id = agent.create_session(session_name=session)
    
    # Create explorer
    explorer = AgentManifoldExplorer(agent)
    
    # Find path
    path = explorer.find_semantic_path(word1, word2, session_id)
    
    if path:
        click.echo(f"Semantic path from {word1} to {word2}:")
        click.echo(" -> ".join(path))
    else:
        click.echo(f"No semantic path found between {word1} and {word2}")

@cli.command()
@click.argument('passage')
@click.option('--model', default='all-MiniLM-L6-v2', help='Model to use for embeddings')
@click.option('--window-size', default=5, help='Window size for analysis')
@click.option('--stride', default=1, help='Stride for sliding window')
def analyze_passage(passage: str, model: str, window_size: int, stride: int):
    """Analyze semantic evolution in a passage."""
    # Get embeddings for all words in passage
    words = passage.split()
    embeddings = get_embeddings(words, model)
    
    # Create analyzer
    analyzer = SemanticTimeSeriesAnalyzer(
        window_size=window_size,
        stride=stride
    )
    
    # Analyze passage
    result = analyzer.analyze_passage(passage, embeddings)
    
    click.echo("Passage Analysis:")
    click.echo(json.dumps(result, indent=2))

@cli.command()
@click.argument('passage')
@click.option('--model', default='meta-llama/Llama-3-70b-chat', help='Llama model to use')
@click.option('--session', default='default', help='Session name')
def agent_analyze_passage(passage: str, model: str, session: str):
    """Analyze passage using Llama Stack agent."""
    # Create agent
    agent = Agent(
        model=model,
        instructions="You are a semantic passage analyzer that helps understand the evolution of meaning in text."
    )
    
    # Create session
    session_id = agent.create_session(session_name=session)
    
    # Create explorer
    explorer = AgentManifoldExplorer(agent)
    
    # Analyze passage
    turn_response = agent.create_turn(
        session_id=session_id,
        messages=[{
            "role": "user",
            "content": f"Analyze the semantic evolution in this passage: {passage}. "
                      f"Identify key semantic shifts, patterns, and relationships. "
                      f"Return the analysis as a JSON object with 'shifts', 'patterns', and 'relationships' fields."
        }]
    )
    
    # Process the response
    for log in AgentEventLogger().log(turn_response):
        if log.type == "output":
            try:
                analysis = json.loads(log.content)
                click.echo("Passage Analysis:")
                click.echo(json.dumps(analysis, indent=2))
            except json.JSONDecodeError:
                click.echo("Failed to parse agent response")

if __name__ == '__main__':
    cli() 