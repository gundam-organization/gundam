---
layout: default
title: FAQs 
next_page: "https://ulyevarou.github.io/GUNDAM-documentation/LineageandLegacy.html"
---

<div id="issues"></div>

<script>
  async function fetchIssues() {
    const response = await fetch('https://github.com/gundam-organization/gundam/issues');
    const issues = await response.json();
    const issuesContainer = document.getElementById('issues');

    issues.forEach(issue => {
      const issueElement = document.createElement('div');
      issueElement.innerHTML = `<h3><a href="${issue.html_url}">${issue.title}</a></h3>`;
      issuesContainer.appendChild(issueElement);
    });
  }

  document.addEventListener('DOMContentLoaded', fetchIssues);
</script>
