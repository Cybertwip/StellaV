package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

const (
	scienceOpenPrefix = "10.14293"
	scienceOpenBase   = "https://www.scienceopen.com"
	crossrefWorks     = "https://api.crossref.org/works"
	scienceOpenAgent  = "StellaV/1.0 (medical-research assistant; mailto:stellav@localhost)"
)

var (
	jatsCloseRE = regexp.MustCompile(`(?i)</jats:p>`)
	htmlTagRE   = regexp.MustCompile(`<[^>]+>`)
)

type ScienceOpenPaper struct {
	DOI      string   `json:"doi"`
	Title    string   `json:"title"`
	Abstract string   `json:"abstract"`
	URL      string   `json:"url"`
	Year     int      `json:"year"`
	Authors  []string `json:"authors"`
	Source   string   `json:"source"`
}

type crossrefWorkList struct {
	Message struct {
		Items []crossrefWork `json:"items"`
	} `json:"message"`
}

type crossrefWork struct {
	DOI      string   `json:"DOI"`
	Title    []string `json:"title"`
	Abstract string   `json:"abstract"`
	URL      string   `json:"URL"`
	Type     string   `json:"type"`
	Author   []struct {
		Given  string `json:"given"`
		Family string `json:"family"`
	} `json:"author"`
	Posted  *crossrefDate `json:"posted"`
	Issued  *crossrefDate `json:"issued"`
	Created *crossrefDate `json:"created"`
}

type crossrefDate struct {
	DateParts [][]int `json:"date-parts"`
}

func scienceOpenURL(doi string) string {
	doi = strings.TrimPrefix(strings.TrimSpace(doi), "https://doi.org/")
	if doi == "" {
		return scienceOpenBase
	}
	return scienceOpenBase + "/hosted-document?doi=" + url.QueryEscape(doi)
}

func stripJATS(text string) string {
	text = jatsCloseRE.ReplaceAllString(text, "\n\n")
	text = htmlTagRE.ReplaceAllString(text, " ")
	return strings.Join(strings.Fields(text), " ")
}

func (w crossrefWork) toPaper() ScienceOpenPaper {
	title := "Untitled preprint"
	if len(w.Title) > 0 && strings.TrimSpace(w.Title[0]) != "" {
		title = strings.TrimSpace(w.Title[0])
	}
	authors := make([]string, 0, len(w.Author))
	for _, a := range w.Author {
		name := strings.TrimSpace(a.Given + " " + a.Family)
		if name != "" {
			authors = append(authors, name)
		}
	}
	year := 0
	for _, d := range []*crossrefDate{w.Posted, w.Issued, w.Created} {
		if d != nil && len(d.DateParts) > 0 && len(d.DateParts[0]) > 0 {
			year = d.DateParts[0][0]
			break
		}
	}
	return ScienceOpenPaper{
		DOI:      strings.TrimSpace(w.DOI),
		Title:    title,
		Abstract: stripJATS(w.Abstract),
		URL:      scienceOpenURL(w.DOI),
		Year:     year,
		Authors:  authors,
		Source:   "scienceopen",
	}
}

func SearchScienceOpen(query string, rows int) ([]ScienceOpenPaper, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	if rows <= 0 {
		rows = 8
	}
	if rows > 25 {
		rows = 25
	}
	params := url.Values{}
	params.Set("query", query)
	params.Set("filter", "prefix:"+scienceOpenPrefix+",type:posted-content,has-abstract:true")
	params.Set("rows", fmt.Sprintf("%d", rows))
	params.Set("select", "DOI,title,author,abstract,type,URL,issued,created")
	params.Set("mailto", "stellav@localhost")
	req, err := http.NewRequest(http.MethodGet, crossrefWorks+"?"+params.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", envOr("STELLAV_USER_AGENT", scienceOpenAgent))
	req.Header.Set("Accept", "application/json")
	client := &http.Client{Timeout: 25 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("scienceopen/crossref HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var payload crossrefWorkList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	out := make([]ScienceOpenPaper, 0, len(payload.Message.Items))
	for _, item := range payload.Message.Items {
		paper := item.toPaper()
		if paper.DOI == "" || paper.Abstract == "" {
			continue
		}
		if !looksMedical(paper.Title + " " + paper.Abstract) {
			continue
		}
		out = append(out, paper)
	}
	return out, nil
}

func looksMedical(text string) bool {
	q := strings.ToLower(text)
	hints := []string{
		"medic", "clinic", "patient", "trial", "disease", "therap", "cancer",
		"cardio", "immun", "infect", "pharma", "genom", "neuro", "epidem",
		"diagnos", "biomarker", "vaccine", "oncolog", "hospital", "health",
	}
	for _, hint := range hints {
		if strings.Contains(q, hint) {
			return true
		}
	}
	return false
}

func defaultScienceOpenQueries() []string {
	return []string{
		"randomized controlled trial",
		"cancer immunotherapy",
		"heart failure outcomes",
		"vaccine immune response",
		"antimicrobial resistance",
		"alzheimer biomarker",
		"pharmacokinetics safety",
		"GWAS precision medicine",
		"public health screening",
		"epidemiology confounding",
	}
}

func indexScienceOpenIntoModel(m *StellaModel, query string, rows int) (int, error) {
	papers, err := SearchScienceOpen(query, rows)
	if err != nil {
		return 0, err
	}
	added := 0
	for _, paper := range papers {
		if m.AddChunk(paper.DOI, paper.Title, paper.Abstract, paper.URL, "scienceopen") {
			added++
		}
	}
	return added, nil
}
