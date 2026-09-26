"""Download annual report PDFs listed in config/companies.yaml."""

import httpx

from filings_agent.config import load_companies, settings

HEADERS = {"User-Agent": "Mozilla/5.0 (filings-agent research project)"}


def download_annual_reports() -> None:
    out_dir = settings.data_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        for company in load_companies():
            for fy, url in company["annual_reports"].items():
                dest = out_dir / f"{company['id']}_{fy}_annual_report.pdf"
                if not url or dest.exists():
                    continue
                print(f"downloading {dest.name}")
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)


if __name__ == "__main__":
    download_annual_reports()
