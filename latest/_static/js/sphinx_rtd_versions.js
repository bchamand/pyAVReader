'use strict';

async function requestURL(url, headers) {
    const init = {
        method: 'GET',
        headers: headers,
    };
    const response = await fetch(url, init);
    if (!response.ok) {
        throw response;
    }
    return response;
}

function getRepoInfo() {
    const res = window.location.href.match(
        /^https?:\/\/([\w_\.-]+)\.github\.io\/([\w_\.-]+)\/v?([\w_\.-]+)/
    );
    if (res === null) {
        return null;
    }
    const [, username, repo, currentVersion] = res;
    return { username, repo, currentVersion };
}

async function findVersions(username, repo) {
    let listVersions;

    try {
        const response = await requestURL(
            `https://${username}.github.io/${repo}/all_versions.txt`,
            new Headers({ 'Content-Type': 'text/plain' })
        );
        listVersions = await response.text();
        listVersions = listVersions.trim().split('\n');
    } catch (error) {
        try {
            const response = await requestURL(
                `https://api.github.com/repos/${username}/${repo}/tags`,
                new Headers({ 'Content-Type': 'application/json' })
            );
            const listTags = await response.json();
            listVersions = listTags.map((value) => value['name']);
        } catch (error) {
            listVersions = null;
        }
    }

    return listVersions;
}

function generateHTMLNode(root, release, listVersions) {
    // create the new element
    const element = document.createElement("div");
    element.className = "rst-versions";
    element.setAttribute("data-toggle", "rst-versions");
    element.setAttribute("role", "note");
    element.setAttribute("aria-label", "versions");

    // generate the html with all the versions
    const html = `
        <span class="rst-current-version" data-toggle="rst-current-version">
            <span class="fa fa-book"> Other versions</span>
            v: ${release.replace(/^v/, '')}
            <span class="fa fa-caret-down"></span>
        </span>
        <div class="rst-other-versions">
            <dl>
                ${listVersions.map(element => `<dd><a href="${root}/${element}/">${element}</a></dd>`).join('')}
            </dl>
        </div>
    `;

    // inject the html into the element
    element.innerHTML = html

    return element;
}

async function onLoad() {
    // retrieve information about the project
    const { username, repo, currentVersion } = getRepoInfo() || {};

    // find all versions and display the version selection bar
    const listVersions = await findVersions(username, repo);

    // change the display name of the version (in case we need to display
    // "stable" or "latest" instead the number version)
    //let version = currentVersion
    if (currentVersion) {
        if (listVersions && (listVersions.indexOf(currentVersion) === 1)) {
            currentVersion = 'stable'
        }
        document.getElementsByClassName('version')[0].innerHTML = currentVersion;
    }

    // generate the version selection bar
    if (listVersions) {
        const html = generateHTMLNode(`/${repo}`, currentVersion, listVersions);
        const element = document.getElementsByClassName('wy-grid-for-nav')[0];
        element.parentNode.insertBefore(html, element.nextSibling);
    }
}

window.addEventListener('load', onLoad)
