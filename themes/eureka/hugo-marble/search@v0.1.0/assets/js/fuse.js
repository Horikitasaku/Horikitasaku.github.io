import Fuse from "js/fuse/fuse.esm.min.js";

const search = () => {
  // get search query
  var url = new URL(window.location.href);
  var q = url.searchParams.get("q");
  var input = document.getElementById("search-input");
  input.value = q;

  // fuse options
  var fuseOptions = {
    includeMatches: true,
    ignoreLocation: true,
    keys: [
      "title",
      "content",
      {{- range $taxonomyName, $taxonomy := .Site.Taxonomies -}}
      "{{- $taxonomyName -}}",
      {{- end -}}
    ],
  };

  // fetch search index
  fetch('{{ "/search/index.json" | absLangURL }}').then((response) => {
    if (response.status !== 200) {
      console.log("Can not fetch search index: " + response.status);
      return;
    }
    response
      .json()
      .then((pages) => {
        var fuse = new Fuse(pages, fuseOptions);
        var results = fuse.search(q);
        if (results.length > 0) {
          populateResults(results);
        } else {
          document.getElementById("search-results").innerHTML =
            '<div id="search-result-empty"><p>No matches found</p></div>';
        }
      })
      .catch((err) => {
        console.log("Failed to search: ", err);
      });
  });
};

const selectHighlightPositions = (positions, k) => {
  // return if length is less than or equal to k
  if (positions.length <= k) {
    return positions;
  }
  positions = findKLargest(positions, 3);
  positions.sort((a, b) => {
    return a[0] > b[0] ? 1 : a[0] < b[0] ? -1 : 0;
  });
  return positions;
};

// find k largest indices in a match
const findKLargest = (positions, k) => {
  let left = 0;
  let right = positions.length - 1;
  while (right - left + 1 >= k) {
    let pivot = Math.floor(Math.random() * (right - left + 1)) + left;
    swap(positions, pivot, right);
    let i = left;
    let j = i;
    while (j < right) {
      if (
        positions[j][1] - positions[j][0] <=
        positions[right][1] - positions[right][0]
      ) {
        swap(positions, i, j);
        i++;
      }
      j++;
    }
    swap(positions, i, right);
    if (right - i + 1 == k) {
      return positions.slice(i, positions.length);
    } else if (right - i + 1 > k) {
      left = i + 1;
    } else {
      k -= right - i + 1;
      right = i - 1;
    }
  }
  return positions.slice(left, positions.length);
};

const swap = (nums, i, j) => ([nums[i], nums[j]] = [nums[j], nums[i]]);

const populateResults = (results) => {
  var searchResults = document.getElementById("search-results");

  // clear previous result
  searchResults.innerHTML = "";

  results.forEach((value, key) => {
    var title = value.item.title;
    var content = value.item.content;
    var permalink = value.item.permalink;

    value.matches.forEach((match) => {
      if (match.key == "title") {
        // handle title highlight
        let origin = title;
        title = "";
        let positions = selectHighlightPositions(match.indices, 3);
        let startIndex = 0;
        positions.forEach((index) => {
          const endIndex = index[1] + 1;
          title += [
            origin.substring(startIndex, index[0]),
            "<mark>",
            origin.substring(index[0], endIndex),
            "</mark>",
          ].join("");
          startIndex = endIndex;
        });
        title += origin.substring(startIndex);
      } else if (match.key == "content") {
        let origin = content;
        content = "";
        // handle content highlight
        let positions = selectHighlightPositions(match.indices, 3);
        positions.sort((a, b) => {
          return a[0] > b[0] ? 1 : a[0] < b[0] ? -1 : 0;
        });
        positions.forEach((position, index) => {
          content += [
            origin.substring(position[0] - 50, position[0]),
            "<mark>",
            origin.substring(position[0], position[1] + 1),
            "</mark>",
            origin.substring(position[1] + 1, position[1] + 51),
          ].join("");
          content += " ... ";
        });
      }
    });

    var output = `<div id="search-result-${key}" class="my-4">
<a href="${permalink}" class="no-underline"><h3 class="mt-0">${title}</h3></a>
<div>${content}</div></div>`;
    searchResults.innerHTML += output;
  });
};

document.addEventListener("DOMContentLoaded", () => {
  search();
});
