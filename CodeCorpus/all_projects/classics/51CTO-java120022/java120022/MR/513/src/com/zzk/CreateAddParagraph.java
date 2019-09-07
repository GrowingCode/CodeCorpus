package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;



public class CreateAddParagraph {
	public static void main(String[] args){
		Document document = new Document();	//创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java编程词典.pdf"));	//关联文档对象与输出流
			document.open();				//打开文档
			Paragraph P1 = new Paragraph("Java programming dictionary");	
			//创建段落并添加内容
			document.add(P1);		//向文档添加段落
			Paragraph P2 = new Paragraph("The richest resource for learning");
			document.add(P2);		//向文档添加段落		
		} catch (FileNotFoundException e) {
				e.printStackTrace();
		} catch (DocumentException e) {
				e.printStackTrace();
		}
		finally{
			document.close();		//关闭文档	
		}
	}
}
