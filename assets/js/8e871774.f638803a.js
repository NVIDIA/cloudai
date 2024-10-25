"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[708],{1954:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>d,contentTitle:()=>o,default:()=>u,frontMatter:()=>i,metadata:()=>l,toc:()=>a});var s=r(4848),t=r(8453);const i={id:"overview"},o="Overview",l={id:"dev/overview",title:"Overview",description:"This document targets developers who want to contribute to the project's core.",source:"@site/../docs/dev/overview.md",sourceDirName:"dev",slug:"/dev/overview",permalink:"/cloudai/docs/dev/overview",draft:!1,unlisted:!1,editUrl:"https://github.com/NVIDIA/cloudai/edit/main/website/../docs/dev/overview.md",tags:[],version:"current",frontMatter:{id:"overview"},sidebar:"dev",previous:{title:"Cloud AI Development",permalink:"/cloudai/docs/dev/"}},d={},a=[{value:"Core Modules",id:"core-modules",level:2},{value:"Runners",id:"runners",level:2},{value:"Installers",id:"installers",level:2},{value:"Systems",id:"systems",level:2}];function c(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",header:"header",mermaid:"mermaid",p:"p",...(0,t.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.header,{children:(0,s.jsx)(n.h1,{id:"overview",children:"Overview"})}),"\n",(0,s.jsx)(n.p,{children:"This document targets developers who want to contribute to the project's core."}),"\n",(0,s.jsx)(n.mermaid,{value:"graph TD\n    subgraph _core\n        base_modules\n        core_implementations\n        registry\n    end\n\n    subgraph runners\n        SlurmRunner\n        StandaloneRunner\n    end\n\n    subgraph installers\n        SlurmInstaller\n        StandaloneInstaller\n    end\n\n    subgraph systems\n        SlurmSystem\n        StandaloneSystem\n    end\n\n    installers --\x3e _core\n    runners --\x3e _core\n    systems --\x3e _core"}),"\n",(0,s.jsx)(n.h2,{id:"core-modules",children:"Core Modules"}),"\n",(0,s.jsxs)(n.p,{children:["We use ",(0,s.jsx)(n.a,{href:"https://github.com/seddonym/import-linter",children:"import-linter"})," to ensure no core modules import higher level modules."]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.code,{children:"Registry"})," object is a singleton that holds implementation mappings. Users can register their own implementations to the registry or replace the default implementations."]}),"\n",(0,s.jsx)(n.h2,{id:"runners",children:"Runners"}),"\n",(0,s.jsx)(n.p,{children:"TBD"}),"\n",(0,s.jsx)(n.h2,{id:"installers",children:"Installers"}),"\n",(0,s.jsx)(n.p,{children:"TBD"}),"\n",(0,s.jsx)(n.h2,{id:"systems",children:"Systems"}),"\n",(0,s.jsx)(n.p,{children:"TBD"})]})}function u(e={}){const{wrapper:n}={...(0,t.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(c,{...e})}):c(e)}}}]);