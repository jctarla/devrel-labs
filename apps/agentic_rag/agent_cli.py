#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from typing import Optional, List
from threading import Thread
import textwrap

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
except ImportError:
    print("Error: 'rich' library is required. Please install it with: pip install rich")
    sys.exit(1)

# Initialize console
console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    title = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                 AGENTIC RAG SYSTEM CLI                         ║
    ║         Oracle AI Vector Search + Ollama (Gemma 3)             ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(Textwrap(title, justify="center"), style="bold cyan"))
    console.print(f"[dim]Working Directory: {os.getcwd()}[/dim]\n")

def Textwrap(text, justify="left"):
    return text # Placeholder if needed, but rich Panel handles string content well

def run_command(command: List[str], description: str = "Processing"):
    """Run a command with a spinner and error handling"""
    console.print(f"[bold green]Running command:[/bold green] {' '.join(command)}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=description, total=None)
            
            # Run command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
        if result.returncode == 0:
            console.print("[bold green]Success![/bold green]")
            if result.stdout:
                console.print(Panel(result.stdout, title="Output", border_style="green"))
            return True
        else:
            console.print("[bold red]Error![/bold red]")
            if result.stderr:
                console.print(Panel(result.stderr, title="Error Details", border_style="red"))
            if result.stdout:
                console.print(Panel(result.stdout, title="Standard Output", border_style="dim"))
            return False
            
    except Exception as e:
        console.print(f"[bold red]Exception occurred:[/bold red] {str(e)}")
        return False

def menu_process_pdfs():
    print_header()
    console.print("[bold yellow]PDF Processor[/bold yellow]")
    console.print("1. Process a single PDF file")
    console.print("2. Process all PDFs in a directory")
    console.print("3. Process a PDF from URL")
    console.print("0. Back to Main Menu")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "3", "0"], default="0")
    
    if choice == "0":
        return
        
    output_file = Prompt.ask("Output JSON file path", default="chunks.json")
    
    if choice == "1":
        input_path = Prompt.ask("Enter path to PDF file")
        if not os.path.exists(input_path):
            console.print(f"[red]File not found: {input_path}[/red]")
            input("Press Enter to continue...")
            return
        run_command(["python", "-m", "src.pdf_processor", "--input", input_path, "--output", output_file], "Processing PDF...")
        
    elif choice == "2":
        input_path = Prompt.ask("Enter directory path")
        if not os.path.isdir(input_path):
            console.print(f"[red]Directory not found: {input_path}[/red]")
            input("Press Enter to continue...")
            return
        run_command(["python", "-m", "src.pdf_processor", "--input", input_path, "--output", output_file], "Processing Directory...")
        
    elif choice == "3":
        input_url = Prompt.ask("Enter PDF URL")
        run_command(["python", "-m", "src.pdf_processor", "--input", input_url, "--output", output_file], "Downloading and Processing PDF...")
    
    input("\nPress Enter to continue...")

def menu_process_websites():
    print_header()
    console.print("[bold yellow]Website Processor[/bold yellow]")
    console.print("1. Process a single website URL")
    console.print("2. Process multiple URLs from a file")
    console.print("0. Back to Main Menu")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "0"], default="0")
    
    if choice == "0":
        return
        
    output_file = Prompt.ask("Output JSON file path", default="docs/web_content.json")
    
    if choice == "1":
        url = Prompt.ask("Enter website URL")
        run_command(["python", "-m", "src.web_processor", "--input", url, "--output", output_file], "Processing Website...")
        
    elif choice == "2":
        input_file = Prompt.ask("Enter URLs file path", default="urls.txt")
        if not os.path.exists(input_file):
            console.print(f"[red]File not found: {input_file}[/red]")
            input("Press Enter to continue...")
            return
        run_command(["python", "-m", "src.web_processor", "--input", input_file, "--output", output_file], "Processing URLs from file...")
    
    input("\nPress Enter to continue...")

def menu_manage_vector_store():
    print_header()
    console.print("[bold yellow]Manage Vector Store[/bold yellow]")
    console.print("1. Add PDF chunks to vector store")
    console.print("2. Add Web chunks to vector store")
    console.print("3. Query vector store directly")
    console.print("0. Back to Main Menu")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "3", "0"], default="0")
    
    if choice == "0":
        return
        
    if choice == "1":
        input_file = Prompt.ask("Enter chunks JSON file", default="chunks.json")
        if not os.path.exists(input_file):
            console.print(f"[red]File not found: {input_file}[/red]")
            input("Press Enter to continue...")
            return
        run_command(["python", "-m", "src.store", "--add", input_file], "Adding PDF chunks...")
        
    elif choice == "2":
        input_file = Prompt.ask("Enter web content JSON file", default="docs/web_content.json")
        if not os.path.exists(input_file):
            console.print(f"[red]File not found: {input_file}[/red]")
            input("Press Enter to continue...")
            return
        run_command(["python", "-m", "src.store", "--add-web", input_file], "Adding Web chunks...")
        
    elif choice == "3":
        query = Prompt.ask("Enter search query")
        run_command(["python", "-m", "src.store", "--query", query], "Querying Vector Store...")
    
    input("\nPress Enter to continue...")

def menu_test_oradb():
    print_header()
    console.print("[bold yellow]Test Oracle DB Connection[/bold yellow]")
    console.print("1. Run basic connection tests")
    console.print("2. Show collection statistics only")
    console.print("3. Run text similarity search")
    console.print("0. Back to Main Menu")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "3", "0"], default="0")
    
    if choice == "0":
        return
        
    if choice == "1":
        # Note: test_oradb.py is in tests/ directory now
        run_command(["python", "tests/test_oradb.py"], "Testing Oracle DB...")
    
    elif choice == "2":
        run_command(["python", "tests/test_oradb.py", "--stats-only"], "Fetching Stats...")
        
    elif choice == "3":
        query = Prompt.ask("Enter test query", default="artificial intelligence")
        run_command(["python", "tests/test_oradb.py", "--query", query], "Running Vector Search...")
        
    input("\nPress Enter to continue...")

def menu_rag_agent():
    while True:
        print_header()
        console.print("[bold yellow]RAG Agent Chat (Gemma 3)[/bold yellow]")
        console.print("Enter your query to chat with the agent.")
        console.print("Type 'exit' or '0' to return to main menu.")
        
        query = Prompt.ask("\n[bold green]Query[/bold green]")
        
        if query.lower() in ['exit', 'quit', '0']:
            break
            
        use_cot = Confirm.ask("Use Chain of Thought reasoning?", default=False)
        
        cmd = ["python", "-m", "src.local_rag_agent", "--query", query]
        if use_cot:
            cmd.append("--use-cot")
            
        run_command(cmd, "Generating Answer...")
        input("\nPress Enter to continue...")

def main_menu():
    while True:
        print_header()
        console.print("[bold]Select a Task:[/bold]")
        
        table = Table(show_header=False, box=None)
        table.add_row("[1]", "Process PDFs", style="cyan")
        table.add_row("[2]", "Process Websites", style="cyan")
        table.add_row("[3]", "Manage Vector Store", style="cyan")
        table.add_row("[4]", "Test Oracle DB", style="cyan")
        table.add_row("[5]", "Chat with Agent (RAG)", style="magenta")
        table.add_row("[0]", "Exit", style="red")
        
        console.print(table)
        
        choice = Prompt.ask("\nEnter choice", choices=["1", "2", "3", "4", "5", "0"], default="5")
        
        if choice == "1":
            menu_process_pdfs()
        elif choice == "2":
            menu_process_websites()
        elif choice == "3":
            menu_manage_vector_store()
        elif choice == "4":
            menu_test_oradb()
        elif choice == "5":
            menu_rag_agent()
        elif choice == "0":
            console.print("[bold]Goodbye![/bold]")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
