"""
Build the UK bird species list with iNaturalist taxon IDs and family data.

Reads the iNat taxa file, matches each species in UK_SPECIES to its taxon_id
and family, then writes data/uk_bird_taxa_full.json.

Usage:
    python training/build_species_list.py
"""

import csv
import gzip
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TAXA_PATH = Path(
    "/Users/liam/Projects/WildlifeAI/data/inat_metadata/taxa.csv.gz"
)
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "uk_bird_taxa_full.json"

# ---------------------------------------------------------------------------
# UK species list — scientific name -> common name
# Taxonomy follows IOC/BTO conventions, scientific names verified against
# the iNaturalist taxa file.
# ---------------------------------------------------------------------------
UK_SPECIES: dict[str, str] = {
    # --- Wildfowl ---
    "Cygnus olor": "Mute Swan",
    "Cygnus columbianus": "Bewick's Swan",
    "Cygnus cygnus": "Whooper Swan",
    "Anser anser": "Greylag Goose",
    "Anser albifrons": "White-fronted Goose",
    "Anser brachyrhynchus": "Pink-footed Goose",
    "Branta canadensis": "Canada Goose",
    "Branta leucopsis": "Barnacle Goose",
    "Branta bernicla": "Brent Goose",
    "Tadorna tadorna": "Shelduck",
    "Anas platyrhynchos": "Mallard",
    "Anas crecca": "Teal",
    "Mareca penelope": "Wigeon",
    "Spatula clypeata": "Shoveler",
    "Mareca strepera": "Gadwall",
    "Aythya fuligula": "Tufted Duck",
    "Aythya ferina": "Pochard",
    "Bucephala clangula": "Goldeneye",
    "Mergus merganser": "Goosander",
    "Mergus serrator": "Red-breasted Merganser",
    "Somateria mollissima": "Eider",
    "Melanitta nigra": "Common Scoter",
    "Aix galericulata": "Mandarin Duck",
    "Oxyura jamaicensis": "Ruddy Duck",
    "Netta rufina": "Red-crested Pochard",
    "Aythya marila": "Scaup",
    "Clangula hyemalis": "Long-tailed Duck",
    # --- Gamebirds ---
    "Phasianus colchicus": "Pheasant",
    "Perdix perdix": "Grey Partridge",
    "Alectoris rufa": "Red-legged Partridge",
    "Lagopus lagopus": "Red Grouse",
    "Lagopus muta": "Ptarmigan",
    "Tetrao urogallus": "Capercaillie",
    "Lyrurus tetrix": "Black Grouse",
    # --- Divers ---
    "Gavia stellata": "Red-throated Diver",
    "Gavia arctica": "Black-throated Diver",
    "Gavia immer": "Great Northern Diver",
    # --- Grebes ---
    "Tachybaptus ruficollis": "Little Grebe",
    "Podiceps cristatus": "Great Crested Grebe",
    "Podiceps nigricollis": "Black-necked Grebe",
    "Podiceps auritus": "Slavonian Grebe",
    "Podiceps grisegena": "Red-necked Grebe",
    # --- Shearwaters & Petrels ---
    "Fulmarus glacialis": "Fulmar",
    "Puffinus puffinus": "Manx Shearwater",
    "Hydrobates pelagicus": "Storm Petrel",
    # --- Gannet & Cormorants ---
    "Morus bassanus": "Gannet",
    "Phalacrocorax carbo": "Cormorant",
    "Gulosus aristotelis": "Shag",
    # --- Herons ---
    "Ardea cinerea": "Grey Heron",
    "Ardea alba": "Great White Egret",
    "Egretta garzetta": "Little Egret",
    "Botaurus stellaris": "Bittern",
    # --- Raptors ---
    "Pandion haliaetus": "Osprey",
    "Milvus milvus": "Red Kite",
    "Milvus migrans": "Black Kite",
    "Haliaeetus albicilla": "White-tailed Eagle",
    "Circus aeruginosus": "Marsh Harrier",
    "Circus cyaneus": "Hen Harrier",
    "Circus pygargus": "Montagu's Harrier",
    "Accipiter nisus": "Sparrowhawk",
    "Accipiter gentilis": "Goshawk",
    "Buteo buteo": "Buzzard",
    "Buteo lagopus": "Rough-legged Buzzard",
    "Aquila chrysaetos": "Golden Eagle",
    "Falco tinnunculus": "Kestrel",
    "Falco columbarius": "Merlin",
    "Falco subbuteo": "Hobby",
    "Falco peregrinus": "Peregrine",
    # --- Rails ---
    "Rallus aquaticus": "Water Rail",
    "Crex crex": "Corncrake",
    "Gallinula chloropus": "Moorhen",
    "Fulica atra": "Coot",
    # --- Waders ---
    "Haematopus ostralegus": "Oystercatcher",
    "Recurvirostra avosetta": "Avocet",
    "Charadrius dubius": "Little Ringed Plover",
    "Charadrius hiaticula": "Ringed Plover",
    "Pluvialis apricaria": "Golden Plover",
    "Pluvialis squatarola": "Grey Plover",
    "Vanellus vanellus": "Lapwing",
    "Calidris canutus": "Knot",
    "Calidris alba": "Sanderling",
    "Calidris pugnax": "Ruff",
    "Calidris alpina": "Dunlin",
    "Calidris maritima": "Purple Sandpiper",
    "Calidris ferruginea": "Curlew Sandpiper",
    "Calidris temminckii": "Temminck's Stint",
    "Calidris minuta": "Little Stint",
    "Scolopax rusticola": "Woodcock",
    "Gallinago gallinago": "Snipe",
    "Lymnocryptes minimus": "Jack Snipe",
    "Limosa limosa": "Black-tailed Godwit",
    "Limosa lapponica": "Bar-tailed Godwit",
    "Numenius phaeopus": "Whimbrel",
    "Numenius arquata": "Curlew",
    "Tringa totanus": "Redshank",
    "Tringa erythropus": "Spotted Redshank",
    "Tringa nebularia": "Greenshank",
    "Tringa ochropus": "Green Sandpiper",
    "Tringa glareola": "Wood Sandpiper",
    "Actitis hypoleucos": "Common Sandpiper",
    "Arenaria interpres": "Turnstone",
    "Phalaropus lobatus": "Red-necked Phalarope",
    # --- Skuas ---
    "Stercorarius skua": "Great Skua",
    "Stercorarius parasiticus": "Arctic Skua",
    # --- Gulls ---
    "Chroicocephalus ridibundus": "Black-headed Gull",
    "Larus canus": "Common Gull",
    "Larus fuscus": "Lesser Black-backed Gull",
    "Larus argentatus": "Herring Gull",
    "Larus marinus": "Great Black-backed Gull",
    "Rissa tridactyla": "Kittiwake",
    "Ichthyaetus melanocephalus": "Mediterranean Gull",
    # --- Terns ---
    "Sterna hirundo": "Common Tern",
    "Sterna paradisaea": "Arctic Tern",
    "Sternula albifrons": "Little Tern",
    "Thalasseus sandvicensis": "Sandwich Tern",
    "Chlidonias niger": "Black Tern",
    # --- Auks ---
    "Uria aalge": "Guillemot",
    "Alca torda": "Razorbill",
    "Fratercula arctica": "Puffin",
    "Cepphus grylle": "Black Guillemot",
    "Alle alle": "Little Auk",
    # --- Pigeons & Doves ---
    "Columba livia": "Rock Dove",
    "Columba oenas": "Stock Dove",
    "Columba palumbus": "Woodpigeon",
    "Streptopelia decaocto": "Collared Dove",
    "Streptopelia turtur": "Turtle Dove",
    # --- Cuckoo ---
    "Cuculus canorus": "Cuckoo",
    # --- Owls ---
    "Tyto alba": "Barn Owl",
    "Strix aluco": "Tawny Owl",
    "Asio otus": "Long-eared Owl",
    "Asio flammeus": "Short-eared Owl",
    "Athene noctua": "Little Owl",
    # --- Nightjar ---
    "Caprimulgus europaeus": "Nightjar",
    # --- Swift ---
    "Apus apus": "Swift",
    # --- Kingfisher ---
    "Alcedo atthis": "Kingfisher",
    # --- Woodpeckers ---
    "Picus viridis": "Green Woodpecker",
    "Dendrocopos major": "Great Spotted Woodpecker",
    "Dryobates minor": "Lesser Spotted Woodpecker",
    # --- Larks ---
    "Alauda arvensis": "Skylark",
    "Lullula arborea": "Woodlark",
    # --- Swallows & Martins ---
    "Hirundo rustica": "Swallow",
    "Delichon urbicum": "House Martin",
    "Riparia riparia": "Sand Martin",
    # --- Pipits & Wagtails ---
    "Anthus pratensis": "Meadow Pipit",
    "Anthus trivialis": "Tree Pipit",
    "Anthus petrosus": "Rock Pipit",
    "Anthus spinoletta": "Water Pipit",
    "Motacilla alba": "Pied Wagtail",
    "Motacilla cinerea": "Grey Wagtail",
    "Motacilla flava": "Yellow Wagtail",
    # --- Waxwing ---
    "Bombycilla garrulus": "Waxwing",
    # --- Dipper ---
    "Cinclus cinclus": "Dipper",
    # --- Wren ---
    "Troglodytes troglodytes": "Wren",
    # --- Dunnock ---
    "Prunella modularis": "Dunnock",
    # --- Chats & Thrushes ---
    "Erithacus rubecula": "Robin",
    "Luscinia megarhynchos": "Nightingale",
    "Phoenicurus ochruros": "Black Redstart",
    "Phoenicurus phoenicurus": "Redstart",
    "Saxicola rubicola": "Stonechat",
    "Saxicola rubetra": "Whinchat",
    "Oenanthe oenanthe": "Wheatear",
    "Turdus merula": "Blackbird",
    "Turdus pilaris": "Fieldfare",
    "Turdus philomelos": "Song Thrush",
    "Turdus iliacus": "Redwing",
    "Turdus viscivorus": "Mistle Thrush",
    "Turdus torquatus": "Ring Ouzel",
    # --- Warblers ---
    "Cettia cetti": "Cetti's Warbler",
    "Locustella naevia": "Grasshopper Warbler",
    "Acrocephalus schoenobaenus": "Sedge Warbler",
    "Acrocephalus scirpaceus": "Reed Warbler",
    "Hippolais icterina": "Icterine Warbler",
    "Sylvia atricapilla": "Blackcap",
    "Sylvia borin": "Garden Warbler",
    "Curruca communis": "Whitethroat",
    "Curruca curruca": "Lesser Whitethroat",
    "Phylloscopus collybita": "Chiffchaff",
    "Phylloscopus trochilus": "Willow Warbler",
    "Phylloscopus sibilatrix": "Wood Warbler",
    "Regulus regulus": "Goldcrest",
    "Regulus ignicapilla": "Firecrest",
    # --- Flycatchers ---
    "Muscicapa striata": "Spotted Flycatcher",
    "Ficedula hypoleuca": "Pied Flycatcher",
    # --- Tits ---
    "Aegithalos caudatus": "Long-tailed Tit",
    "Poecile palustris": "Marsh Tit",
    "Poecile montanus": "Willow Tit",
    "Periparus ater": "Coal Tit",
    "Cyanistes caeruleus": "Blue Tit",
    "Parus major": "Great Tit",
    "Lophophanes cristatus": "Crested Tit",
    # --- Nuthatch & Treecreeper ---
    "Sitta europaea": "Nuthatch",
    "Certhia familiaris": "Treecreeper",
    # --- Shrikes ---
    "Lanius collurio": "Red-backed Shrike",
    "Lanius excubitor": "Great Grey Shrike",
    # --- Corvids ---
    "Garrulus glandarius": "Jay",
    "Pica pica": "Magpie",
    "Nucifraga caryocatactes": "Nutcracker",
    "Pyrrhocorax pyrrhocorax": "Chough",
    "Coloeus monedula": "Jackdaw",
    "Corvus frugilegus": "Rook",
    "Corvus corone": "Carrion Crow",
    "Corvus cornix": "Hooded Crow",
    "Corvus corax": "Raven",
    # --- Starling ---
    "Sturnus vulgaris": "Starling",
    # --- Sparrows ---
    "Passer domesticus": "House Sparrow",
    "Passer montanus": "Tree Sparrow",
    # --- Finches ---
    "Fringilla coelebs": "Chaffinch",
    "Fringilla montifringilla": "Brambling",
    "Chloris chloris": "Greenfinch",
    "Carduelis carduelis": "Goldfinch",
    "Spinus spinus": "Siskin",
    "Linaria cannabina": "Linnet",
    "Linaria flavirostris": "Twite",
    "Acanthis flammea": "Redpoll",
    "Acanthis cabaret": "Lesser Redpoll",
    "Loxia curvirostra": "Crossbill",
    "Loxia scotica": "Scottish Crossbill",
    "Pyrrhula pyrrhula": "Bullfinch",
    "Coccothraustes coccothraustes": "Hawfinch",
    # --- Buntings ---
    "Emberiza citrinella": "Yellowhammer",
    "Emberiza cirlus": "Cirl Bunting",
    "Emberiza schoeniclus": "Reed Bunting",
    "Emberiza calandra": "Corn Bunting",
    "Plectrophenax nivalis": "Snow Bunting",
    "Calcarius lapponicus": "Lapland Bunting",
    # --- Misc ---
    "Upupa epops": "Hoopoe",
    "Merops apiaster": "Bee-eater",
    "Psittacula krameri": "Ring-necked Parakeet",
}

# ---------------------------------------------------------------------------
# Family display names — scientific family -> human-readable display name
# ---------------------------------------------------------------------------
FAMILY_DISPLAY: dict[str, str] = {
    # Waterfowl
    "Anatidae": "Wildfowl",
    # Gamebirds
    "Phasianidae": "Gamebirds",
    "Odontophoridae": "New World Quails",
    # Divers
    "Gaviidae": "Divers",
    # Grebes
    "Podicipedidae": "Grebes",
    # Petrels & Shearwaters
    "Procellariidae": "Petrels & Shearwaters",
    "Hydrobatidae": "Storm Petrels",
    "Oceanitidae": "Southern Storm Petrels",
    # Gannet & Boobies
    "Sulidae": "Gannets & Boobies",
    # Cormorants
    "Phalacrocoracidae": "Cormorants",
    # Herons
    "Ardeidae": "Herons & Egrets",
    # Raptors
    "Pandionidae": "Osprey",
    "Accipitridae": "Hawks & Eagles",
    "Falconidae": "Falcons",
    # Rails
    "Rallidae": "Rails & Coots",
    # Waders
    "Haematopodidae": "Oystercatchers",
    "Recurvirostridae": "Avocets & Stilts",
    "Charadriidae": "Plovers",
    "Scolopacidae": "Sandpipers & Allies",
    "Phalaropodidae": "Phalaropes",
    # Skuas
    "Stercorariidae": "Skuas",
    # Gulls & Terns
    "Laridae": "Gulls & Terns",
    # Auks
    "Alcidae": "Auks",
    # Pigeons
    "Columbidae": "Pigeons & Doves",
    # Cuckoo
    "Cuculidae": "Cuckoos",
    # Owls
    "Tytonidae": "Barn Owls",
    "Strigidae": "Owls",
    # Nightjar
    "Caprimulgidae": "Nightjars",
    # Swift
    "Apodidae": "Swifts",
    # Kingfisher
    "Alcedinidae": "Kingfishers",
    # Bee-eater
    "Meropidae": "Bee-eaters",
    # Hoopoe
    "Upupidae": "Hoopoes",
    # Woodpeckers
    "Picidae": "Woodpeckers",
    # Larks
    "Alaudidae": "Larks",
    # Swallows
    "Hirundinidae": "Swallows & Martins",
    # Pipits & Wagtails
    "Motacillidae": "Pipits & Wagtails",
    # Waxwing
    "Bombycillidae": "Waxwings",
    # Dipper
    "Cinclidae": "Dippers",
    # Wren
    "Troglodytidae": "Wrens",
    # Dunnock
    "Prunellidae": "Accentors",
    # Chats, Flycatchers, Robins
    "Muscicapidae": "Old World Flycatchers",
    # Thrushes
    "Turdidae": "Thrushes",
    # Warblers
    "Cettiidae": "Bush Warblers",
    "Locustellidae": "Grasshopper Warblers",
    "Acrocephalidae": "Reed Warblers",
    "Sylviidae": "Sylvia Warblers",
    "Currucidae": "Whitethroats",
    "Phylloscopidae": "Leaf Warblers",
    "Hippolaididae": "Hippolais Warblers",
    "Icteridae": "Icterine & Melodious Warblers",
    # Kinglets / Crests
    "Regulidae": "Crests",
    # Tits
    "Aegithalidae": "Long-tailed Tits",
    "Paridae": "Tits",
    # Nuthatch
    "Sittidae": "Nuthatches",
    # Treecreeper
    "Certhiidae": "Treecreepers",
    # Shrikes
    "Laniidae": "Shrikes",
    # Corvids
    "Corvidae": "Crows & Jays",
    # Starling
    "Sturnidae": "Starlings",
    # Sparrows
    "Passeridae": "Old World Sparrows",
    # Finches
    "Fringillidae": "Finches",
    # Buntings & Longspurs
    "Emberizidae": "Buntings",
    "Calcariidae": "Buntings & Longspurs",
    # Parakeets
    "Psittaculidae": "Parakeets",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_taxa(taxa_path: Path) -> dict[str, dict]:
    """Load the gzipped taxa CSV into a dict keyed by taxon_id string."""
    taxa: dict[str, dict] = {}
    with gzip.open(taxa_path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            taxa[row["taxon_id"]] = row
    log.info("Loaded %d taxa records", len(taxa))
    return taxa


def find_family(
    taxon_id: str, all_taxa: dict[str, dict]
) -> tuple[str, str] | tuple[None, None]:
    """Walk the ancestry chain to find the family-rank ancestor.

    Returns (scientific_family_name, scientific_family_name) on success,
    or (None, None) if no family ancestor is found.
    """
    taxon = all_taxa.get(taxon_id)
    if not taxon:
        return None, None

    ancestry = taxon.get("ancestry", "")
    if not ancestry:
        return None, None

    ancestor_ids = ancestry.split("/")
    # Walk from most-specific to least-specific ancestor
    for anc_id in reversed(ancestor_ids):
        anc = all_taxa.get(anc_id)
        if anc and anc.get("rank") == "family":
            name = anc["name"]
            return name, name

    return None, None


def main() -> None:
    if not TAXA_PATH.exists():
        log.error("Taxa file not found: %s", TAXA_PATH)
        sys.exit(1)

    all_taxa = load_taxa(TAXA_PATH)

    # Build name -> list of taxa for active species-rank entries first, then
    # fall back to inactive entries (some species are inactive in iNat but
    # still have valid records we can use).
    name_to_taxon: dict[str, dict] = {}
    name_to_taxon_inactive: dict[str, dict] = {}

    for taxon in all_taxa.values():
        if taxon.get("rank") != "species":
            continue
        name = taxon["name"]
        if taxon.get("active") == "true":
            # Prefer the active entry; keep first encountered
            if name not in name_to_taxon:
                name_to_taxon[name] = taxon
        else:
            if name not in name_to_taxon_inactive:
                name_to_taxon_inactive[name] = taxon

    results: list[dict] = []
    missing: list[str] = []

    for scientific_name, common_name in UK_SPECIES.items():
        taxon = name_to_taxon.get(scientific_name) or name_to_taxon_inactive.get(
            scientific_name
        )
        if not taxon:
            log.warning("No taxon found for: %s (%s)", scientific_name, common_name)
            missing.append(scientific_name)
            continue

        taxon_id = taxon["taxon_id"]
        sci_family, _ = find_family(taxon_id, all_taxa)

        if sci_family is None:
            log.warning(
                "No family found for %s (taxon_id=%s)", scientific_name, taxon_id
            )
            sci_family = "Unknown"

        display_family = FAMILY_DISPLAY.get(sci_family, sci_family)

        if taxon.get("active") != "true":
            log.warning(
                "Using inactive taxon for %s (%s, id=%s)",
                scientific_name,
                common_name,
                taxon_id,
            )

        results.append(
            {
                "common_name": common_name,
                "scientific_name": scientific_name,
                "taxon_id": taxon_id,
                "scientific_family": sci_family,
                "display_family": display_family,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    log.info(
        "Wrote %d species to %s", len(results), OUTPUT_PATH
    )
    if missing:
        log.warning(
            "%d species not found in taxa file: %s",
            len(missing),
            ", ".join(missing),
        )
    else:
        log.info("All %d species resolved successfully", len(UK_SPECIES))

    print(f"\nFound: {len(results)}/{len(UK_SPECIES)}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
