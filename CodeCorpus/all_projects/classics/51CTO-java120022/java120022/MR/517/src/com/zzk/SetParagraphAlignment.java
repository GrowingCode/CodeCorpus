package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;


public class SetParagraphAlignment {
	
	
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Paragraph paragraph1 = new Paragraph("www.mingrosoft.com");// 创建段落并添加内容
			paragraph1.setAlignment(Element.ALIGN_LEFT); // 左对齐
			document.add(new Paragraph(paragraph1));     // 向文档添加段落	
			Paragraph paragraph2 = new Paragraph("www.mingribook.com");// 创建段落并添加内容
			paragraph2.setAlignment(Element.ALIGN_RIGHT);// 右对齐
			document.add(new Paragraph(paragraph2));	 // 向文档添加段落
					
			Paragraph paragraph3 = new Paragraph("www.mingribccd.com"); // 创建段落并添加内容
			paragraph3.setAlignment(Element.ALIGN_CENTER);// 居中对齐 
			document.add(new Paragraph(paragraph3));	  // 向文档添加段落	
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	 }
}

