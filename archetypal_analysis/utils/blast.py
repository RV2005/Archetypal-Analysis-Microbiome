"""
BLAST analysis utilities for taxonomic identification.  UNTESTED
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from Bio import Blast


def parse_blast_xml(xml_source: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse BLAST XML output and extract hit information.
    
    Args:
        xml_source: XML content as string
        
    Returns:
        Dictionary with query sequences as keys and lists of hit dictionaries as values
    """
    taxa_hits = {}
    
    # Parse XML
    root = ET.fromstring(xml_source)
    
    # Find all iterations (queries)
    for iteration in root.findall(".//Iteration"):
        query_def = iteration.find("Iteration_query-def").text
        taxa_hits[query_def] = []
        
        # Extract hits for this query
        for hit in iteration.findall(".//Iteration_hits/Hit"):
            hit_dict = {}
            
            # Extract hit information
            for child in hit:
                try: 
                    hit_dict[child.tag] = float(child.text) 
                except (ValueError, TypeError):
                    hit_dict[child.tag] = child.text
                
                # Handle HSP (High-scoring Segment Pair) data
                if child.tag == "Hit_hsps":
                    hit_dict["Hsps"] = {}
                    hsps = child.findall(".//Hsp")[0].findall("./")
                    for hsp in hsps:
                        try: 
                            hit_dict["Hsps"][hsp.tag] = float(hsp.text)
                        except (ValueError, TypeError):
                            hit_dict["Hsps"][hsp.tag] = hsp.text
            
            taxa_hits[query_def].append(hit_dict)
    
    return taxa_hits


def write_fasta(taxa_list: List[str], output_path: str, append: bool = True) -> None:
    """
    Write taxa names to FASTA format file.
    
    Args:
        taxa_list: List of taxa names to write
        output_path: Path to output FASTA file
        append: Whether to append to existing file
    """
    mode = "a" if append else "w"
    
    with open(output_path, mode) as out_stream:
        for taxa in taxa_list:
            if taxa and taxa != "Others":
                out_stream.write(f">{taxa}\n")
                out_stream.write(f"{taxa}\n")


def run_blast_query(fasta_file: str, 
                   blast_db: str = "ref_prok_rep_genomes",
                   blast_program: str = "blastn",
                   output_file: str = "blast_results.xml") -> str:
    """
    Run BLAST query online and save results.
    
    Args:
        fasta_file: Path to input FASTA file
        blast_db: BLAST database to search
        blast_program: BLAST program to use
        output_file: Path to save XML results
        
    Returns:
        Path to output XML file
    """
    # Read FASTA content
    with open(fasta_file, "r") as in_stream:
        fasta_content = in_stream.read()
    
    # Run BLAST query
    result_stream = Blast.qblast(blast_program, blast_db, fasta_content)
    
    # Save results
    with open(output_file, "wb") as out_stream:
        out_stream.write(result_stream.read())
    
    result_stream.close()
    return output_file


def filter_blast_results(taxa_hits: Dict[str, List[Dict[str, Any]]], 
                        num_hits: int = 3,
                        sort_by: str = "Hsp_evalue") -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter and sort BLAST results.
    
    Args:
        taxa_hits: Results from parse_blast_xml
        num_hits: Number of top hits to keep per query
        sort_by: Field to sort hits by (lower is better)
        
    Returns:
        Filtered results dictionary
    """
    filtered_hits = {}
    
    for taxa, hits in taxa_hits.items():
        # Sort hits by specified field (assuming lower is better)
        if hits and sort_by in hits[0].get("Hsps", {}):
            sorted_hits = sorted(hits, key=lambda x: x["Hsps"].get(sort_by, float('inf')))
        else:
            sorted_hits = hits
        
        # Keep top N hits
        filtered_hits[taxa] = sorted_hits[:num_hits]
    
    return filtered_hits


def get_best_hit_descriptions(taxa_hits: Dict[str, List[Dict[str, Any]]], 
                             max_label_len: int = 25) -> Dict[str, str]:
    """
    Get the best hit description for each query taxa.
    
    Args:
        taxa_hits: Results from parse_blast_xml or filter_blast_results
        max_label_len: Maximum length for description labels
        
    Returns:
        Dictionary mapping taxa to their best hit descriptions
    """
    best_descriptions = {}
    
    for taxa, hits in taxa_hits.items():
        if hits:
            # Get the best hit (first in sorted list)
            best_hit = hits[0]
            description = best_hit.get("Hit_def", taxa)
            
            # Truncate if too long
            if len(description) > max_label_len:
                description = description[:max_label_len]
            
            best_descriptions[taxa] = description
        else:
            best_descriptions[taxa] = taxa
    
    return best_descriptions


class BlastAnalyzer:
    """High-level interface for BLAST analysis of archetypal compositions."""

    def __init__(
        self,
        blast_db: str = "ref_prok_rep_genomes",
        blast_program: str = "blastn",
        num_hits: int = 3,
    ):
        """
        Initialize BLAST analyzer.

        Args:
            blast_db: BLAST database to use
            blast_program: BLAST program to use
            num_hits: Number of hits to retain per query
        """
        self.blast_db = blast_db
        self.blast_program = blast_program
        self.num_hits = num_hits

    def analyze_archetype_taxa(
        self,
        top_taxa_archs: List[List[Tuple[float, str]]],
        temp_fasta: str = "temp_archetype.fasta",
        output_xml: str = "archetype_blast.xml",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze taxa from archetype compositions using BLAST.

        Args:
            top_taxa_archs: List of archetype compositions from get_archetype_composition
            temp_fasta: Temporary FASTA file path
            output_xml: Output XML file path

        Returns:
            BLAST results dictionary
        """
        # Extract unique taxa (excluding "Others")
        unique_taxa = set()
        for arch in top_taxa_archs:
            for val, taxa in arch:
                if taxa != "Others":
                    unique_taxa.add(taxa)

        # Write to FASTA
        write_fasta(list(unique_taxa), temp_fasta, append=False)

        # Run BLAST
        run_blast_query(temp_fasta, self.blast_db, self.blast_program, output_xml)

        # Parse results
        with open(output_xml, "r") as f:
            xml_content = f.read()

        taxa_hits = parse_blast_xml(xml_content)

        # Filter results
        filtered_hits = filter_blast_results(taxa_hits, self.num_hits)

        return filtered_hits

    def get_annotated_compositions(
        self,
        top_taxa_archs: List[List[Tuple[float, str]]],
        blast_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        **blast_kwargs
    ) -> List[List[Tuple[float, Any]]]:
        """
        Get archetype compositions with BLAST annotations.

        Args:
            top_taxa_archs: Original archetype compositions
            blast_results: Pre-computed BLAST results (optional)
            **blast_kwargs: Arguments for analyze_archetype_taxa

        Returns:
            List of archetype compositions with BLAST hit information
        """
        if blast_results is None:
            blast_results = self.analyze_archetype_taxa(top_taxa_archs, **blast_kwargs)

        annotated_archs = []

        for arch in top_taxa_archs:
            annotated_arch = []
            for val, taxa in arch:
                if taxa == "Others":
                    annotated_arch.append((val, taxa))
                else:
                    # Replace taxa with BLAST results
                    hits = blast_results.get(taxa, [])
                    annotated_arch.append((val, hits))
            annotated_archs.append(annotated_arch)

        return annotated_archs
